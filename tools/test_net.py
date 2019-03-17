import argparse
import os
import logging
import datetime
import json

import torch

from image_captioning.config import cfg
from image_captioning.data import make_data_loader
from image_captioning.solver import make_optimizer
from image_captioning.solver import make_lr_scheduler
from image_captioning.engine.trainer import do_train
from image_captioning.engine.inference import inference
from image_captioning.modeling.decoder.build import build_decoder
from image_captioning.utils.checkpoint import ModelCheckpointer
from image_captioning.utils.get_vocab import get_vocab
from image_captioning.utils.collect_env import collect_env_info
from image_captioning.utils.logger import setup_logger
from image_captioning.utils.comm import get_rank
from image_captioning.utils.miscellaneous import mkdir
from image_captioning.modeling.utils import LanguageModelCriterion
from image_captioning.utils.comm import synchronize, is_main_process


def test(cfg, verbose=False, distributed=False):
    logger = logging.getLogger('image_captioning.test')
    dataset = cfg.DATASET.TEST
    vocab = get_vocab(dataset)
    device = cfg.MODEL.DEVICE
    decoder = build_decoder(cfg, vocab)
    checkpointer = ModelCheckpointer(cfg, decoder)
    checkpointer.load(cfg.MODEL.WEIGHT)
    decoder = decoder.to(device)
    test_data_loder = make_data_loader(
        cfg,
        split='test',
        is_distributed=distributed
    )
    criterion = LanguageModelCriterion()
    beam_size = cfg.TEST.BEAM_SIZE
    loss, predictions, scores = inference(
        decoder,
        criterion,
        test_data_loder,
        dataset,
        vocab,
        beam_size,
        device
    )
    if is_main_process():
        now = datetime.datetime.now()
        file_name = os.path.join(cfg.OUTPUT_DIR, "test-" + now.strftime("%Y%m%d-%H%M%S") + ".json")
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
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER
    )
    parser.add_argument(
        '--verbose',
        help="show test results",
        action="store_true"
    )

    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = setup_logger('image_captioning', cfg.OUTPUT_DIR, get_rank(), "testing_log.txt")
    logger.info("Using {} GPUs.".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    if args.config_file:
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = '\n' + cf.read()
            logger.info(config_str)
    test(cfg, args.verbose, args.distributed)


if __name__ == '__main__':
    main()
