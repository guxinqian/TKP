from __future__ import absolute_import

from .ResNet import *

__factory = {
    'vid_nonlocalresnet50': VidNonLocalResNet50,
    'img_resnet50': ImgResNet50,
    'classifier': Classifier,
}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
