import argparse
import json

from torch.utils.data import DataLoader
import torch

from models import *
from utils.datasets import *
from utils.utils import *

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

def test(
        imgs_path=None,
        labels_path=None,
        saved_model = None,
        names = None,
        batch_size=16,
        img_size=416,
        iou_thres=0.5,
        conf_thres=0.001,
        nms_thres=0.5,
        save_json=False,
        model=None,
        dataloader = None,
        debug = False,
):
    if model is None:
        cuda = torch.cuda.is_available()
        device = torch.device('cuda:0' if cuda else 'cpu')

        # Initialize model
        model = torch.load(saved_model).to(device)

    else:
        device = next(model.parameters()).device  # get model device

    # change model hyp
    model.hyp = hyp

    # Configure run
    class_names = load_classes(names)  # class names
    nc = len(class_names)


    # Dataloader
    if dataloader is None:
        dataset = LoadImagesAndLabels(imgs_path,labels_path, img_size=img_size, debug=debug)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                # num_workers=4,
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)


    seen = 0
    model.eval()
    # coco91class = coco80_to_coco91_class()
    print(('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1'))
    loss, p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0., 0.
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc='Computing mAP')):
        targets = targets.to(device)
        imgs = imgs.to(device)

        # Plot images with bounding boxes
        # if batch_i == 0 and not os.path.exists('test_batch0.jpg'):
        #     plot_images(imgs=imgs, targets=targets, fname='test_batch0.jpg')

        # Run model
        inf_out, train_out = model(imgs)  # inference and training outputs

        # Compute loss
        if hasattr(model, 'hyp'):  # if model has loss hyperparameters
            loss_i, _ = compute_loss(train_out, targets, model)
            loss += loss_i.item()

        # Run NMS
        output = non_max_suppression(inf_out, conf_thres=conf_thres, nms_thres=nms_thres)

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split('_')[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(imgs[si].shape[1:], box, shapes[si])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for di, d in enumerate(pred):
                    jdict.append({
                        'image_id': image_id,
                        'category_id': int(d[6]),
                        'bbox': [float3(x) for x in box[di]],
                        'score': float(d[4])
                    })

                with open('results.json', 'w') as file:
                    json.dump(jdict, file)

            # Assign all predictions as incorrect
            correct = [0] * len(pred)
            if nl:
                detected = []
                tbox = xywh2xyxy(labels[:, 1:5]) * img_size  # target boxes

                # Search for correct predictions
                for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                    # Break if all targets already located in image
                    if len(detected) == nl:
                        break

                    # Continue if predicted class not among image classes
                    if pcls.item() not in tcls:
                        continue

                    # Best iou, index between pred and targets
                    iou, bi = bbox_iou(pbox, tbox).max(0)

                    # If iou > threshold and class is correct mark as correct
                    if iou > iou_thres and bi not in detected:  # and pcls == tcls[bi]:
                        correct[i] = 1
                        detected.append(bi)

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
    nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1), end='\n\n')

    # Print results per class
    if nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (class_names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Return results
    return mp, mr, map, mf1, loss / len(dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--imgs_path', type=str,help='file path contain images')
    parser.add_argument('--labels_path', type=str, help='file path contain labels')
    parser.add_argument('--names', type=str, help='file path contain class names')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--save-json',type=int, default=1, help='save a cocoapi-compatible JSON results file: 0 False, 1 True')
    parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--debug', type = int, default= 0, help='Whether to use two images to check: 0 False, 1 True')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    with torch.no_grad():
        mAP = test(
            imgs_path = opt.imgs_path,
            labels_path = opt.labels_path,
            names = opt.names,
            weights = opt.weights,
            batch_size = opt.batch_size,
            img_size = opt.img_size,
            iou_thres = opt.iou_thres,
            conf_thres = opt.conf_thres,
            nms_thres = opt.nms_thres,
            save_json = True if opt.save_json else False,
            debug = True if opt.debug else False,
        )
