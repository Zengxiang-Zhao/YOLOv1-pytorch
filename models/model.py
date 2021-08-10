from torchvision import  models
import torch.nn as nn
from collections import OrderedDict
import torch

def build_model(anchors,nc,img_size):
    resnet50 = models.resnet50(pretrained=True)
    model_backbone = nn.Sequential(
        OrderedDict([
                     ('conv1', resnet50.conv1),
                     ('bn1', resnet50.bn1),
                     ('relu', resnet50.relu),
                     ('maxpool',resnet50.maxpool),
                     ('layer1', resnet50.layer1),
                     ('layer2', resnet50.layer2),
                     ('layer3', resnet50.layer3),
                     ('layer4',resnet50.layer4),
        ]))
    model_neck = nn.Sequential(nn.Conv2d(2048,3*(5+1),kernel_size=3,stride=1,padding=1))
    model_detector = Detector(anchors,nc,img_size=img_size)
    model = nn.Sequential(OrderedDict([
         ('backbone', model_backbone),
         ('neck', model_neck),
         ('detector', model_detector)                          
        ]))
    return model



class Detector(nn.Module):
    def __init__(self, anchors, nc, img_size=416):
        super(Detector, self).__init__()

        self.anchors = torch.Tensor(anchors)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.nx = 0  # initialize number of x gridpoints
        self.ny = 0  # initialize number of y gridpoints
        self.img_size = img_size

    def forward(self, p):

        bs, ny, nx = p.shape[0], p.shape[-2], p.shape[-1]
        if (self.ny, self.nx) != (ny, nx):
            create_grids(self, self.img_size, (ny, nx), p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.nc + 5, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        else:  # inference
            io = p.clone()  # inference output
            io[..., 0:2] = torch.sigmoid(io[..., 0:2]) + self.grid_xy  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            # io[..., 2:4] = ((torch.sigmoid(io[..., 2:4]) * 2) ** 3) * self.anchor_wh  # wh power method
            io[..., 4:] = torch.sigmoid(io[..., 4:])  # p_conf, p_cls
            # io[..., 5:] = F.softmax(io[..., 5:], dim=4)  # p_cls
            io[..., :4] *= self.stride

            # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
            return io.view(bs, -1, 5 + self.nc), p

def create_grids(self, img_size, ng, device='cpu'):
    ny, nx = ng  # x and y grid size
    self.img_size = img_size
    self.stride = img_size / max(ng)

    # build xy offsets
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    self.grid_xy = torch.stack((xv, yv), 2).to(device).float().view((1, 1, ny, nx, 2))

    # build wh gains
    self.anchor_vec = self.anchors.to(device) / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).to(device)
    self.ng = torch.Tensor(ng).to(device)
    self.nx = nx
    self.ny = ny