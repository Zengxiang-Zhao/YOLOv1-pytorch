import os
import numpy as np
def parse_cfg(cfgfile):
    """
    Purpose: parse the yolov1.cfg file and return list of dictionary
    Args:
        cfgfile: yolov1.cfg
    Return:
        list of dictinonary : [layer0, layer1]
    """
    with open(cfgfile) as f:
        cfgs = f.read()

    cfg_lines = cfgs.split('\n')
    # get ride of blank lines and empty string on both sides of the line
    cfg_lines = [l.strip() for l in cfg_lines if len(l) > 0 ] 
    layer_list = []
    layer = {}
    flag = True #  whether this is the first layer

    for l in cfg_lines:
        if l[0] == '[': # encourter a new layer
            if not flag:
                layer_list.append(layer) 
                layer = {}
                
            layer['type'] = l.strip()[1:-1]
            flag = False
                
        elif l[0] == '#':
            continue
        else:
            key,value = l.split('=')
            layer[key]=value

    layer_list.append(layer)
    return layer_list



