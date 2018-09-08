import os
import torch, torchvision
import models
from models import vgg
from models import densenet
from models import resnet

# map between model name and function
models = {
    'vgg9'                  : models.vgg.VGG9,
    'resnet18'              : models.resnet.ResNet18,
    'resnet18_noshort'      : models.resnet.ResNet18_noshort,
    'resnet34'              : models.resnet.ResNet34,
    'resnet34_noshort'      : models.resnet.ResNet34_noshort,
    'resnet50'              : models.resnet.ResNet50,
    'resnet50_noshort'      : models.resnet.ResNet50_noshort,
    'resnet101'             : models.resnet.ResNet101,
    'resnet101_noshort'     : models.resnet.ResNet101_noshort,
    'resnet152'             : models.resnet.ResNet152,
    'resnet152_noshort'     : models.resnet.ResNet152_noshort,
    'resnet20'              : models.resnet.ResNet20,
    'resnet20_noshort'      : models.resnet.ResNet20_noshort,
    'resnet32_noshort'      : models.resnet.ResNet32_noshort,
    'resnet44_noshort'      : models.resnet.ResNet44_noshort,
    'resnet50_16_noshort'   : models.resnet.ResNet50_16_noshort,
    'resnet56'              : models.resnet.ResNet56,
    'resnet56_noshort'      : models.resnet.ResNet56_noshort,
    'resnet110'             : models.resnet.ResNet110,
    'resnet110_noshort'     : models.resnet.ResNet110_noshort,
    'wrn56_2'               : models.resnet.WRN56_2,
    'wrn56_2_noshort'       : models.resnet.WRN56_2_noshort,
    'wrn56_4'               : models.resnet.WRN56_4,
    'wrn56_4_noshort'       : models.resnet.WRN56_4_noshort,
    'wrn56_8'               : models.resnet.WRN56_8,
    'wrn56_8_noshort'       : models.resnet.WRN56_8_noshort,
    'wrn110_2_noshort'      : models.resnet.WRN110_2_noshort,
    'wrn110_4_noshort'      : models.resnet.WRN110_4_noshort,
}

def load(model_name, model_file=None, data_parallel=False):
    net = models[model_name]()
    if data_parallel: # the model is saved in data paralle mode
        net = torch.nn.DataParallel(net)

    if model_file:
        stored = torch.load(model_file, map_location=lambda storage, loc: storage)
        if 'state_dict' in stored.keys():
            net.load_state_dict(stored['state_dict'])
        else:
            net.load_state_dict(stored)

    if data_parallel: # convert the model back to the single GPU version
        net = net.module

    net.eval()
    return net
