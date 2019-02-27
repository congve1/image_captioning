import json
import pickle
import os
import logging
import argparse
from collections import defaultdict

from image_captioning.config import cfg
from image_captioning.utils.imports import import_file
from image_captioning.utils.logger import setup_logger
from image_captioning.utils.comm import get_rank
from image_captioning.utils.get_vocab import get_vocab

def precook(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1,n+1):
      for i in range(len(words)-k+1):
        ngram = tuple(words[i:i+k])
        counts[ngram] += 1
    return counts


def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
    """Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    """
    return [precook(ref, n) for ref in refs]


def create_crefs(refs):
    crefs = []
    for ref in refs:
        #  ref is a list of 5 captions
        crefs.append(cook_refs(ref))
    return crefs


def compute_doc_freq(crefs):
    """
    Compute term frequency for reference data.
    This will be used to compute idf (inverse document frequency later)
    The term frequency is stored in the object
    :return: None
    """
    document_frequency = defaultdict(float)
    for refs in crefs:
        # refs, k ref captions of one image
        for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
            document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)
    return document_frequency


def generate_ngrams(vocab):
    paths_catalog = import_file(
            'image_captioning.config.paths_catalog', cfg.PATHS_CATALOG, True
        )
    DatasetCatalog = paths_catalog.DatasetCatalog
    logger = logging.getLogger("image_captioning.build_ngrams")
    for dataset, split in [(cfg.DATASET.TRAIN, 'train')]:
        data = DatasetCatalog.get(dataset)
        ann_file = data['args']['ann_file']
        logger.info("load annotation file {}".format(ann_file))
        with open(ann_file, 'r') as f:
            ann_file = json.load(f)
        refs_words = []
        refs_idxs = []
        count_imgs = 0
        logger.info("start processing captions")
        for image in ann_file['images']:
            if(split == image['split']) or \
                    (split == 'train' and image['split'] in ['train', 'restval']):
                ref_words = []
                ref_idxs = []
                for sent in image['sentences']:
                    tmp_tokens = sent['tokens'] + ['<end>']
                    tmp_tokens = [
                        token if vocab[token] != vocab['<unk>']
                        else '<unk>'
                        for token in tmp_tokens
                    ]
                    ref_words.append(' '.join(tmp_tokens))
                    ref_idxs.append(' '.join([str(vocab[token]) for token in tmp_tokens]))
                refs_idxs.append(ref_idxs)
                refs_words.append(ref_words)
                count_imgs += 1
                if count_imgs % 100 == 0:
                   logger.info("Processed {}/{} images."
                               .format(count_imgs, len(ann_file['images'])))
        ngram_words = compute_doc_freq(create_crefs(refs_words))
        ngram_idxs = compute_doc_freq(create_crefs(refs_idxs))
        ngram_words_path = os.path.join(data['args']['root'], dataset+'_words.pkl')
        ngram_idxs_path = os.path.join(data['args']['root'], dataset+'_idxs.pkl')
        logger.info("save n-gram words to {}".format(ngram_words_path))
        with open(ngram_words_path, 'wb') as f:
            pickle.dump(
                {'document_frequency': ngram_words, 'ref_len': count_imgs},
                f, protocol=pickle.HIGHEST_PROTOCOL
            )
        logger.info('save n-gram indexes to {}'.format(ngram_idxs_path))
        with open(ngram_idxs_path, 'wb') as f:
            pickle.dump(
                {'document_frequency': ngram_idxs, 'ref_len': count_imgs},
                f, protocol=pickle.HIGHEST_PROTOCOL
            )

def main():
    logger = setup_logger('image_captioning', cfg.OUTPUT_DIR, get_rank(), "generating_ngrams_log.txt")
    vocab = get_vocab("coco_2014")
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
    if args.config_file:
        cfg.merge_from_file(args.config_file)
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = '\n' + cf.read()
            logger.info(config_str)
    cfg.merge_from_list(args.opts)
    generate_ngrams(vocab)


if __name__ == "__main__":
    main()


