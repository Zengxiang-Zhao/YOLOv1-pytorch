from models.darknet import build_modules

cfgfile = './cfg/yolov1.cfg'

module_list, layers = build_modules(cfgfile)

print('Done')