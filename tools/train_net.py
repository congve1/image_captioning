import argparse
import os
import logging

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
from image_captioning.utils.collect_env import  collect_env_info
from image_captioning.utils.logger import setup_logger
from image_captioning.utils.miscellaneous import mkdir
from image_captioning.modeling.utils import LanguageModelCriterion
from image_captioning.utils.rewards import init_scorer
from image_captioning.utils.imports import import_file


def train(cfg):
    vocab = get_vocab(cfg.DATASET.TRAIN)
    paths_catalog = import_file(
            'image_captioning.config.paths_catalog', cfg.PATHS_CATALOG, True
        )
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset = DatasetCatalog.get(cfg.DATASET.TRAIN)
    if cfg.SOLVER.SCST_AFTER != 1:
        cached_tokens = os.path.join(dataset['args']['root'], cfg.DATASET.TRAIN+"_words.pkl")
        init_scorer(cached_tokens)
    model = build_decoder(cfg, vocab)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    arguments = dict()
    arguments['iteration'] = 0
    arguments['best_cider_score'] = -10000

    output_dir = cfg.OUTPUT_DIR
    save_to_disk = True
    checkpointer = ModelCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    train_data_loader = make_data_loader(
        cfg,
        start_iter=arguments['iteration']
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    log_period = cfg.SOLVER.LOG_PERIOD
    val_period = cfg.SOLVER.VAL_PERIOD

    do_train(
        model,
        train_data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        log_period,
        val_period,
        val,
        vocab,
        arguments,
    )
    return model


def val(model, device):
    dataset_name = cfg.DATASET.VAL
    vocab = get_vocab(dataset_name)
    val_data_loder = make_data_loader(
        cfg,
        split='val'
    )
    criterion = LanguageModelCriterion()
    return inference(
        model,
        criterion,
        val_data_loder,
        dataset_name,
        vocab,
        cfg.MODEL.DECODER.BEAM_SIZE,
        device
    )


def test(cfg, model):
    logger = logging.getLogger("image_captioning.test")
    device = torch.device(cfg.MODEL.DEVICE)
    dataset_name = cfg.DATASET.TEST
    vocab = get_vocab(dataset_name)
    criterion = LanguageModelCriterion()
    test_data_loader = make_data_loader(
        cfg,
        split='test'
    )
    beam_size = cfg.TEST.BEAM_SIZE
    test_loss, predictions, scores = inference(
        model,
        criterion,
        test_data_loader,
        dataset_name,
        vocab,
        beam_size,
        device
    )
    for prediction in predictions:
        image_id = prediction['image_id']
        caption = prediction['caption']
        logger.info(
            "image_id:{}\nsent:{}".format(
                image_id, caption
            )
        )
    for metric, score in scores.items():
        logger.info(
            "metirc {}, score: {:.4f}".format(
                metric, score
            )
        )


def main():
    parser = argparse.ArgumentParser(description='Pytorch image captioning trainging')
    parser.add_argument(
        '--config-file',
        default='',
        metavar="FILE",
        help='path to config file',
        type=str,
    )
    parser.add_argument(
        '--skip-test',
        dest='skip_test',
        help="Do not test the final model",
        action='store_true',
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
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)
    logger = setup_logger('image_captioning', output_dir)
    logger.info(args)

    #logger.info("Collecting env info (might take some time)")
    #logger.info("\n" + collect_env_info())

    if args.config_file:
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = '\n' + cf.read()
            logger.info(config_str)
    model = train(cfg)
    if not args.skip_test:
        test(cfg, model)


if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn', force=True)
    main()
