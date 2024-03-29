import argparse
import time
import os
import math
import re

import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

import test_map  # Import test.py to get mAP after each epoch
from models.model import Resnetmodel
from utils.datasets import *
from utils.utils import *

# Hyperparameters
# 0.861      0.956      0.936      0.897       1.51      10.39     0.1367    0.01057    0.01181     0.8409     0.1287   0.001028     -3.441     0.9127  0.0004841
hyp = {'k': 10.39,  # loss multiple
       'xy': 0.1367,  # xy loss fraction
       'wh': 0.01057,  # wh loss fraction
       'cls': 0.01181,  # cls loss fraction
       'conf': 0.8409,  # conf loss fraction
       'iou_t': 0.1287,  # iou target-anchor training threshold
       'lr0': 0.001028,  # initial learning rate
       'lrf': -3.441,  # final learning rate = lr0 * (10 ** lrf)
       'momentum': 0.9127,  # SGD momentum
       'weight_decay': 0.0004841,  # optimizer weight decay
       }


def train(
        train_imgs_path,
        train_labels_path,
        img_size=416,
        epochs=273,  # 500200 batches at bs 64, dataset length 117263
        batch_size=16,
        transfer = False,
        debug = False,
        names = None,
        test_imgs_path=None,
        test_labels_path=None,
        resume = False,
        model = None,
        save = False,
        weights = None,
        adam = None,
        eps = 0.0,
):
    weights = weights + os.sep
    latest = weights + 'latest.pt'
    best = weights + 'best.pt'

    class_names = load_classes(names)
    nc = len(class_names)
    anchors = [[116, 90], [156, 198], [373, 326]]

    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    # Initialize model
    if model == None:
        model = Resnetmodel(anchors,nc,img_size).to(device)
    else:
        model = model.to(device)

    # Optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2 += [v]  # biases
        elif re.search(r'conv.*weight',k) != None:
            pg1 += [v]  # apply weight_decay
        elif re.search('neck.*weight',k) != None:
            pg1 += [v]
        else:
            pg0 += [v]  # all else
    # Optimizer
    if adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'])
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    # optimizer = optim.Adam(model.parameters(),lr = hyp['lr0'])
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print('Optimizer groups: %g .bias, %g Conv2d.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_loss = float('inf')
    nf = 255  # yolo layer size (i.e. 255)



    # Scheduler https://github.com/ultralytics/yolov3/issues/238

    # lf = lambda x: 1 - 10 ** (hyp['lrf'] * (1 - x / epochs))  # inverse exp ramp
    # # lf = lambda epoch: 0.65 ** epoch

    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=start_epoch - 1)

    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1
    # Dataset for train
    train_dataset = LoadImagesAndLabels(train_imgs_path, train_labels_path, img_size=img_size, augment=True, debug=debug)

    print(f'There are totally {len(train_dataset)} images to process!')

    # Dataloader for train
    train_dataloader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            # num_workers=opt.num_workers,
                            shuffle=True,
                            pin_memory=True,
                            collate_fn=train_dataset.collate_fn)

    # for test 
    test_dataset = LoadImagesAndLabels(test_imgs_path,test_labels_path, img_size=img_size, debug=debug)
    test_dataloader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            # num_workers=4,
                            pin_memory=True,
                            collate_fn=test_dataset.collate_fn)

    # Start training
    t = time.time()
    model.hyp = hyp  # attach hyperparameters to model
    model.nc = nc
    # model_info(model)

    nb = len(train_dataloader)
    results = (0, 0, 0, 0, 0)  # P, R, mAP, F1, test_loss

    print(f'There are totally **{len(train_dataset)}** images to process for training!')
    print(f'There are totally **{len(test_dataset)}** images to process for testing!')

    for epoch in range(start_epoch, epochs):
        model.train()
        print(('\n%8s%12s' + '%10s' * 7) % ('Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'nTargets', 'time'))

        mloss = torch.zeros(5).to(device)  # mean losses
        for i, (imgs, targets, _, _) in enumerate(train_dataloader):
            imgs = imgs.to(device)
            targets = targets.to(device)
            nt = len(targets)

            optimizer.zero_grad()
            # Run model
            pred = model(imgs)

            # optimizer.zero_grad()
            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model,eps)
            if torch.isnan(loss):
                print('WARNING: nan loss detected, ending training')
                return results

            loss.backward()
            optimizer.step()
            scheduler.step()
            # optimizer.zero_grad()

            # Update running mean of tracked metrics
            mloss = (mloss * i + loss_items) / (i + 1)

            # Print batch results
            s = ('%8s%12s' + '%10.3g' * 7) % (
                '%g/%g' % (epoch, epochs - 1),
                '%g/%g' % (i, nb - 1), *mloss, nt, time.time() - t)
            t = time.time()
            print(s)

        with torch.no_grad():
            print('\n')
            results = test_map.test(names = names, batch_size=batch_size, img_size=img_size, model=model,
                                conf_thres=0.1, dataloader=test_dataloader)
        # Write epoch results
        with open('results.txt', 'a') as file:
            file.write(s + '%11.3g' * 5 % results + '\n')  # P, R, mAP, F1, test_loss

        # Update best loss
        test_loss = results[4]
        if test_loss < best_loss:
            best_loss = test_loss

        # Save training results
        if save:
            # Create checkpoint
            chkpt = {'epoch': epoch,
                     'best_loss': best_loss,
                     'model': model.state_dict(),
                     'optimizer': optimizer.state_dict()}

            # Save latest checkpoint
            torch.save(chkpt, latest)

            # Save best checkpoint
            if best_loss == test_loss:
                torch.save(chkpt, best)

            # # Save backup every 10 epochs (optional)
            # if epoch > 0 and epoch % 10 == 0:
            #     torch.save(chkpt, weights + 'backup%g.pt' % epoch)

            # Delete checkpoint
            del chkpt

    return results



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=273, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--weights', type=str, help='to save weights in the weights folder')
    parser.add_argument('--names', type=str, help='file conatin object names')
    parser.add_argument('--img-size', type=int, default=416, help='pixels')
    parser.add_argument('--train_imgs_path', type=str, help='folder contain images')
    parser.add_argument('--train_labels_path', type=str, help='folder contain labels')
    parser.add_argument('--test_imgs_path', type=str, help='folder contain images')
    parser.add_argument('--test_labels_path', type=str, help='folder contain labels')
    parser.add_argument('--transfer', type = int,default=0, help='Whether only train the yolo layers: 0 False, 1 True')
    parser.add_argument('--debug', type=int,default=0, help='if Ture only use two images: 0 False, 1 True')
    parser.add_argument('--resume', type=int,default=0, help='if Ture to resume: 0 False, 1 True')
    parser.add_argument('--eps', type=str,default=0.0, help='for smoothing lables in compute loss')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    # Train
    results = train(
        train_imgs_path = opt.train_imgs_path,
        train_labels_path = opt.train_labels_path,
        test_imgs_path = opt.test_imgs_path,
        test_labels_path = opt.test_labels_path,
        img_size=opt.img_size,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        transfer= True if opt.transfer else False,
        debug = True if opt.debug else False,
        names = opt.names,
        resume = True if opt.resume else False,
        weights = opt.weights,
        eps = float(opt.eps)
    )


