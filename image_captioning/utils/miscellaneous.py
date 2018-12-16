import errno
import os


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def encode_caption(vocab, caption):
    """
    :param vocab: vocabulary stores word_to_ix dict
    :param caption: list of tokens, first token must be '<start>', last token must be '<end>'
    """
    encoded_caption = []
    for token in caption:
        encoded_caption.append(vocab[token])
    return encoded_caption


def decode_sequence(vocab, seq):
    num_captions, length = seq.shape
    out = []
    for cap_ix in range(num_captions):
        txt = ""
        for token_ix in range(length):
            ix = seq[cap_ix, token_ix].item()
            if vocab[ix] == '<end>':
                break
            if vocab[ix] != '<pad>':
                if token_ix >= 1:
                    txt = txt + ' '
                txt = txt + vocab[ix]
        out.append(txt)
    return out




