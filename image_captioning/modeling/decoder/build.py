import image_captioning.modeling.decoder.models
from image_captioning.modeling import registry


def build_decoder(cfg, vocab):
    assert cfg.MODEL.DECODER.ARCH in registry.DECODER_MODELS, \
        "cfg.MODEL.DECODER.ARCH: {} is not registered in registry".format(
            cfg.MODEL.DECODER.ARCH
        )
    return registry.DECODER_MODELS[cfg.MODEL.DECODER.ARCH](cfg, vocab)




