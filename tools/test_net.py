import argparse
import os
import json

import PIL.Image as Image
import torch

from image_captioning.config import cfg
from image_captioning.utils.imports import import_file
from image_captioning.utils.get_vocab import get_vocab
from image_captioning.modeling.encoder import build_encoder
from image_captioning.modeling.decoder import build_decoder
from image_captioning.data.transforms.build import  build_transforms
from image_captioning.utils.miscellaneous import decode_sequence
from image_captioning.utils.checkpoint import ModelCheckpointer
from image_captioning.utils.logger import  setup_logger


def main():
    """
    This is only use to generate coco online test results
    Returns:

    """
    parser = argparse.ArgumentParser(description='Pytorch image captioning trainging')
    parser.add_argument(
        '--config-file',
        default='',
        metavar="FILE",
        help='path to config file',
        type=str,
    )
    parser.add_argument(
        '--alg-name',
        default='clwupc',
        help='your algorithm name',
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
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    logger = setup_logger("image_captiong")
    device = cfg.MODEL.DEVICE
    beam_size = cfg.MODEL.DECODER.BEAM_SIZE
    paths_catalog = import_file(
        "image_captioning.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    ResNetCatlalog = paths_catalog.ResNetCatalog
    # get data set
    dataset_val = DatasetCatalog.get(cfg.DATASET.VAL)
    dataset_test = DatasetCatalog.get(cfg.DATASET.TEST)
    vocab = get_vocab(cfg.DATASET.VAL)
    # build encoder
    encoder = build_encoder(cfg)
    encoder = encoder.to(device)
    # build decoder
    decoder = build_decoder(cfg, vocab)
    checkpointer = ModelCheckpointer(cfg, decoder)
    checkpointer.load(cfg.MODEL.WEIGHT)
    decoder = decoder.to(device)
    # set to eval mode
    encoder.eval()
    decoder.eval()
    # generate val results
    generate_results_json(
        dataset_val['args']['root'],
        dataset_val['args']['ann_file'],
        vocab,
        encoder,
        decoder,
        device,
        beam_size,
        os.path.join(
            cfg.OUTPUT_DIR, 'captions_val2014_'+args.alg_name+'_results.json'
        ),
        logger,
    )
    # generate test results
    generate_results_json(
        dataset_test['args']['root'],
        dataset_test['args']['ann_file'],
        vocab,
        encoder,
        decoder,
        device,
        beam_size,
        os.path.join(
            cfg.OUTPUT_DIR,
            'captions_test2014_'+args.alg_name+'_results.json'
        ),
        logger,
    )


def generate_results_json(
    img_dir, ann_file, vocab, encoder, decoder,
    device, beam_size, output_json, logger
):
    logger.info("Loading annotation file {}".format(ann_file))
    with open(ann_file, 'r') as f:
        ann_file = json.load(f)
    descriptions = []
    transform = build_transforms()
    logger.info("Start generating captions.")
    with torch.no_grad():
        for idx, img in enumerate(ann_file['images']):
            img_file = os.path.join(img_dir, img['file_name'])
            img_t = Image.open(img_file).convert('RGB')
            img_t = transform(img_t).to(device)
            fc_feature, att_feature = encoder(img_t.unsqueeze(0))
            seq, probs, weights = decoder.decode_search(
                fc_feature, att_feature, beam_size
            )
            sentecne = decode_sequence(vocab, seq)
            entry = {'image_id': img['id'], 'caption': sentecne[0]}
            logger.info("processed {}/{}".format(
                idx+1, len(ann_file['images'])
            ))
            descriptions.append(entry)
    with open(output_json, 'w') as f:
        json.dump(descriptions, f)


if __name__ == '__main__':
    main()