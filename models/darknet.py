import os
import torch
import torch.nn as nn
from utils.util import parse_cfg

def build_modules(cfgfile):
    """
    create nn.ModuleList based on cfgfile

    """
    layers = parse_cfg(cfgfile)

    filter_nums = [3] # to store the former CNN filter numbers, 3 means the original image has 3 channels
    module_list = nn.ModuleList()
    # the first layer is net info
    for i, layer in enumerate(layers[1:]):
        module = nn.Sequential()
        module_type = layer['type']

        if module_type == "convolutional":
            filters = int(layer['filters'])
            size= int(layer['size'])
            stride= int(layer['stride'])
            pad= int(layer['pad'])
            if pad:
                padding = (size-1) // 2
            else:
                padding = 0

            activation=layer['activation']
            bn = int(layer['batch_normalize'])


            conv = nn.Conv2d(in_channels=filter_nums[-1], 
                out_channels=filters,
                kernel_size=size,
                stride=stride,
                padding=padding,
                bias = not bn,
                )
            module.add_module('conv-%d'%i, conv)

            if bn:
                module.add_module('batch_normalize-%d'%i, nn.BatchNorm2d(filters))
            if activation == 'leaky':
                module.add_module('activation-%d'%i, nn.LeakyReLU())

        elif module_type == 'maxpool':
            size=int(layer['size'])
            stride=int(layer['stride'])
            module.add_module('maxpool-%d'%i, nn.MaxPool2d(kernel_size=size,stride=stride))

        elif module_type == 'local': 
            size=int(layer['size'])
            stride=int(layer['stride'])
            pad= int(layer['pad'])
            if pad:
                padding = (size-1)//2
            else:
                padding = 0

            filters= int(layer['filters'])
            activation=layer['activation']
            conv = nn.Conv2d(in_channels=filter_nums[-1],
                out_channels=filters,
                kernel_size = size,
                stride = stride,
                padding = padding,
                bias = True,
                )
            module.add_module('local-%d'%i, conv)

            if activation == 'leaky':
                module.add_module('activation-%d'%i, nn.LeakyReLU())

        elif module_type == 'dropout':
            flatten = nn.Flatten()
            probability = float(layer['probability'])
            module.add_module('flatten-%d'%i, flatten)
            module.add_module('dropout-%d'%i, nn.Dropout2d(p=probability))

        elif module_type == 'connected':
            output= int(layer['output'])
            activation=layer['activation']
            linear_1 = nn.Linear(in_features= 12544, out_features=output)
            linear_2 = nn.Linear(in_features=output,out_features=output)

            module.add_module('connected-%d-1'%i, linear_1)
            module.add_module('connected-%d-2'%i, linear_2)

        elif module_type == 'detection':
            classes=int(layer['classes'])
            coords=int(layer['coords'])
            rescore=int(layer['rescore'])
            side=int(layer['side'])
            num=int(layer['num'])
            softmax=int(layer['softmax'])
            sqrt=int(layer['sqrt'])
            jitter=float(layer['jitter'])

            object_scale=float(layer['object_scale'])
            noobject_scale=float(layer['noobject_scale'])
            class_scale=float(layer['class_scale'])
            coord_scale=float(layer['coord_scale'])

            detection_layer = Detection(classes,coords, rescore,side, num)
            module.add_module('detection-%d'%i, detection_layer)

        module_list.append(module)
        filter_nums.append(filters)

    return module_list, layer

class Detection(nn.Module):
    def __init__(self,classes, coords, rescore, side, num):
        super(Detection,self).__init__()
        self.classes = classes
        self.coords = coords
        self.rescore = rescore
        self.side = side
        self.num = num

    def forward(self, x):
        self.bs = x.shape[0]
        x = x.resized(self.bs,self.side, self.side,(self.num*self.coords + self.classes))

        return x





