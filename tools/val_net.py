import argparse
import os
import logging
import json
import datetime

import torch

from image_captioning.config import cfg
from image_captioning.data import make_data_loader
from image_captioning.solver import make_optimizer
from image_captioning.solver import make_lr_scheduler
from image_captioning.engine.trainer import do_train
from image_captioning.engine.inference import inference
from image_captioning.modeling.decoder import build_decoder
from image_captioning.utils.checkpoint import ModelCheckpointer
from image_captioning.utils.get_vocab import get_vocab
from image_captioning.utils.collect_env import collect_env_info
from image_captioning.utils.logger import setup_logger
from image_captioning.utils.miscellaneous import mkdir
from image_captioning.modeling.utils import LanguageModelCriterion


def val(cfg, verbose=False):
    logger = logging.getLogger('image_captioning.val')
    dataset = cfg.DATASET.VAL
    vocab = get_vocab(dataset)
    device = cfg.MODEL.DEVICE
    decoder = build_decoder(cfg,vocab)
    checkpointer = ModelCheckpointer(cfg, decoder)
    checkpointer.load(cfg.MODEL.WEIGHT)
    decoder = decoder.to(device)
    val_data_loder = make_data_loader(
        cfg,
        split='val'
    )
    criterion = LanguageModelCriterion()
    beam_size = cfg.TEST.BEAM_SIZE
    loss, predictions, scores = inference(
        decoder,
        criterion,
        val_data_loder,
        dataset,
        vocab,
        beam_size,
        device
    )
    now = datetime.datetime.now()
    file_name = os.path.join(cfg.OUTPUT_DIR, "val-"+now.strftime("%Y%m%d-%H%M%S")+".json")
    json.dump(predictions, open(file_name, 'w'))
    logger.info("save results to {}".format(file_name))
    for metric, score in scores.items():
        logger.info(
            "metric {}: {:.4f}".format(
                metric, score
            )
        )
    if verbose:
        for pred in predictions:
            logger.info("image id:{}\ncaption:{}".format(
                pred['image_id'], pred['caption']
            ))



def main():
    parser = argparse.ArgumentParser(description='Pytorch image captioning validating')
    parser.add_argument(
        '--config-file',
        default='',
        metavar="FILE",
        help='path to config file',
        type=str,
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER
    )
    parser.add_argument(
        '--verbose',
        help='show val results',
        action='store_true'
    )

    args = parser.parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = setup_logger('image_captioning')
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    if args.config_file:
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = '\n' + cf.read()
            logger.info(config_str)
    val(cfg, args.verbose)


if __name__ == '__main__':
    main()

