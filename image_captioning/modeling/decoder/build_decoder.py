from .decoder import  Decoder


def build_decoder(cfg, vocab):
    return Decoder(cfg, vocab)