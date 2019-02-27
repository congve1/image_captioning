import pickle
from collections import Counter
import pickle
import json
import argparse
import logging

from image_captioning.config import cfg
from image_captioning.utils.imports import import_file
from image_captioning.data.vocab import Vocab
from image_captioning.utils.logger import setup_logger
from image_captioning.utils.comm import get_rank
from image_captioning.utils.miscellaneous import unicode_to_ascii


def build_vocab():
    logger = logging.getLogger("image_captioning")
    paths_catalog = import_file(
            'image_captioning.config.paths_catalog', cfg.PATHS_CATALOG, True
        )
    DatasetCatalog = paths_catalog.DatasetCatalog
    data = DatasetCatalog.get('coco_2014')
    vocab_file = data['vocab_file']
    ann_file = data['args']['ann_file']
    word_freq = Counter()
    logger.info("loading annotation file {}".format(ann_file))
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    for img in data['images']:
        #if img['split'] in ['train', 'restval']:
        for sentence in img['sentences']:
            word_freq.update(sentence['tokens'])
    logger.info("start building vocab")
    word_to_idx = {
        '<pad>': 0,
        '<start>': 1,
        '<end>': 2,
        '<unk>': 3
    }
    idx = len(word_to_idx) 
    for word, word_count in word_freq.items():
        #word = unicode_to_ascii(word)
        if not word:
            continue
        if word in word_to_idx.keys():
            continue
        if word_count >= cfg.VOCAB.WORD_COUNT_THRESHOLD:
            word_to_idx[word] = idx
            idx += 1
    idx_to_word = {v: k for k, v in word_to_idx.items()} 
    logger.info("vocab building succeeds. vocab size: {}".format(len(word_to_idx)))
    vocab = Vocab(word_to_idx, idx_to_word)
    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab, f)
    logger.info("saved vocab to file {}".format(vocab_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-file',
        help='conifguration file that contains dataset names',
        default='',
        type=str
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER
    )
    args = parser.parse_args()
    logger = setup_logger('image_captioning', cfg.OUTPUT_DIR, get_rank(), "vocab_log.txt")
    logger.info("merge options from list {}".format(args.opts))
    if args.config_file:
        cfg.merge_from_file(args.config_file)
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = '\n' + cf.read()
            logger.info(config_str)
    cfg.merge_from_list(args.opts)
    build_vocab()
