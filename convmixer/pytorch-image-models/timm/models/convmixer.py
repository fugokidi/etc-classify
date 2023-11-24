import sys
sys.path.append('/home/maung/Projects/convmixer/')
from convmixer import ConvMixer, ConvMixer2
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model
import torch.nn as nn


_cfg = {
    'url': '',
    'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
    'crop_pct': .96, 'interpolation': 'bicubic',
    'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head'
}


@register_model
def convmixer_1536_20(pretrained=False, **kwargs):
    model = ConvMixer(1536, 20, kernel_size=9, patch_size=7, n_classes=1000)
    model.default_cfg = _cfg
    return model

# @register_model
# def convmixer_768_32(pretrained=False, **kwargs):
#     model = ConvMixer(768, 32, kernel_size=7, patch_size=7, n_classes=1000)
#     model.default_cfg = _cfg
#     return model

@register_model
def convmixer_768_32(pretrained=False, **kwargs):
    model = ConvMixer(384, 8, kernel_size=9, patch_size=4, n_classes=1000)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_768_16(pretrained=False, **kwargs):
    # kernel_size 9 works better
    model = ConvMixer(384, 8, kernel_size=9, patch_size=4, num_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_384_8(pretrained=False, **kwargs):
    model = ConvMixer(384, 8, kernel_size=9, patch_size=4, num_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_147_16(pretrained=False, **kwargs):
    # kernel_size 9 works better
    model = ConvMixer(147, 16, kernel_size=9, patch_size=7, num_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_768_20(pretrained=False, **kwargs):
    model = ConvMixer(768, 20, kernel_size=9, patch_size=16, num_classes=10)
    model.default_cfg = _cfg
    return model


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
# CIFAR10_STD = (0.2023, 0.1994, 0.2010)

_cfg_cifar10 = {
    'url': '',
    'num_classes': 10, 'input_size': (3, 32, 32), 'pool_size': None,
    'crop_pct': .96, 'interpolation': 'bicubic',
    'mean': CIFAR10_MEAN, 'std': CIFAR10_STD, 'classifier': 'head'
}


@register_model
def tiny_convmixer(pretrained=False, **kwargs):
    model = ConvMixer(256, 8, kernel_size=9, patch_size=1, num_classes=10)
    model.default_cfg = _cfg_cifar10
    return model

@register_model
def c10_convmixer(pretrained=False, **kwargs):
    model = ConvMixer(256, 8, kernel_size=9, patch_size=1, num_classes=10)
    model.default_cfg = _cfg_cifar10
    return model

@register_model
def convmixer2(pretrained=False, **kwargs):
    model = ConvMixer2(48, 16, kernel_size=9, patch_size=1, num_classes=10)
    model.default_cfg = _cfg_cifar10
    return model
