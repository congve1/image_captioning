import errno
import os
import unicodedata
import string

all_letters = string.ascii_letters+"<>"
def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def unicode_to_ascii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
        and c in all_letters
    )

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
    """
    change tensor of word indexes to string
    Args:
        vocab (Voab):  the vocabulary contains the  mapping between word and index
        seq (torch.(cuda).LongTensor): size (num_seqs, seq_length). The sequence
        may contain <end> word

    Returns:
        out (list): a list of string that doesn't contain any <end> or <pad> words
    """
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




