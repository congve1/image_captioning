from collections import OrderedDict

from torch import nn

from image_captioning.modeling import registry

from .import resnet


@registry.ENCODERS.register('R-50-C4')
@registry.ENCODERS.register('R-50-C5')
@registry.ENCODERS.register('R-101-C4')
@registry.ENCODERS.register('R-101-C5')
def build_resnet_encoder(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    return model


def build_encoder(cfg):
    assert cfg.MODEL.ENCODER.CONV_BODY in registry.ENCODERS,\
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.ENCODER.CONV_BODY
        )
    return registry.ENCODERS[cfg.MODEL.ENCODER.CONV_BODY](cfg)
