from image_captioning.modeling import registry


def build_decoder_core(cfg, vocab):
    import image_captioning.modeling.decoder.decoder_core
    assert cfg.MODEL.DECODER.CORE in registry.DECODER_CORES,\
        "cfg.MODEL.DECODER.CORE: {} is not registered in registry".format(
        cfg.MODEL.DECODER.CORE
    )
    return registry.DECODER_CORES[cfg.MODEL.DECODER.CORE](cfg, vocab)


def build_decoder_attention(cfg):
    import image_captioning.modeling.decoder.decoder_atttion
    assert cfg.MODEL.DECODER.ATTENTION in registry.DECODER_ATTENTIONS, \
        "cfg.MODEL.DECODER.ATTENTION: {} is not registered in registry".format(
        cfg.MODEL.DECODER.ATTENTION
    )
    return registry.DECODER_ATTENTIONS[cfg.MODEL.DECODER.ATTENTION](cfg)


