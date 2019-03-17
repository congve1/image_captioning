from image_captioning.modeling import registry

from .base_decoder import BaseDecoder
from .att_decoder import AttDecoder
from .decoder_core import build_decoder_core


@registry.DECODER_MODELS.register("Baseline")
class FCModel(BaseDecoder):
    def __init__(self, cfg, vocab):
        super(FCModel, self).__init__(cfg, vocab)
        self.core = build_decoder_core(cfg, vocab, "Baseline")


@registry.DECODER_MODELS.register("TopDown")
class TopDownModel(AttDecoder):
    def __init__(self, cfg, vocab):
        super(TopDownModel, self).__init__(cfg, vocab)
        self.core = build_decoder_core(cfg, vocab, "TopDownCore")


@registry.DECODER_MODELS.register("ConvHidden")
class ConvHiddenModel(AttDecoder):
    def __init__(self, cfg, vocab):
        super(ConvHiddenModel, self).__init__(cfg, vocab)
        self.core = build_decoder_core(cfg, vocab, "ConvHiddenCore")


@registry.DECODER_MODELS.register("TopDownChannel")
class ChannelModel(AttDecoder):
    def __init__(self, cfg, vocab):
        super(ChannelModel, self).__init__(cfg, vocab)
        self.core = build_decoder_core(cfg, vocab, "ChannelCore")

