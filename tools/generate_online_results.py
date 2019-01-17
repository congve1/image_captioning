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
from image_captioning.modeling.utils import cat


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
    logger = setup_logger("image_captioning")
    device = cfg.MODEL.DEVICE
    beam_size = cfg.TEST.BEAM_SIZE
    paths_catalog = import_file(
        "image_captioning.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    ResNetCatalog = paths_catalog.ResNetCatalog
    if cfg.DATASET.VAL:
        vocab = get_vocab(cfg.DATASET.VAL)
    elif cfg.DATASET.TEST:
        vocab = get_vocab(cfg.DATASET.TEST)
    else:
        raise ValueError("no dataset specified")
    # build encoder
    encoder = build_encoder(cfg)
    encoder_loader = ModelCheckpointer(cfg, encoder)
    url = ResNetCatalog.get(cfg.MODEL.ENCODER.CONV_BODY)
    encoder_loader.load(url)
    encoder = encoder.to(device)
    # build decoder
    decoder = build_decoder(cfg, vocab)
    decoder_loader = ModelCheckpointer(cfg, decoder)
    decoder_loader.load(cfg.MODEL.WEIGHT)
    decoder = decoder.to(device)
    # set to eval mode
    encoder.eval()
    decoder.eval()
    split = ['val', 'test']
    for idx, dataset_name in enumerate([cfg.DATASET.VAL, cfg.DATASET.TEST]):
        if not dataset_name:
            continue
        logger.info('load dataset {}'.format(dataset_name))
        dataset = DatasetCatalog.get(dataset_name)
        generate_results_json(
            dataset['args']['root'],
            dataset['args']['ann_file'],
            vocab,
            encoder,
            decoder,
            device,
            beam_size,
            os.path.join(
                cfg.OUTPUT_DIR, 'captions_'+split[idx]+"_2014_"+args.alg_name+'_results.json'
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
    batch_size = cfg.TEST.IMS_PER_BATCH
    num_imgs = len(ann_file['images'])
    images = ann_file['images']
    logger.info("Start generating captions.")
    with torch.no_grad():
        for idx in range(0, num_imgs, batch_size):
            idx_end = idx + batch_size if (idx+batch_size) < num_imgs else num_imgs
            image_list = []
            for i in range(idx, idx_end):
                img_file = os.path.join(img_dir, images[i]['file_name'])
                img_t = Image.open(img_file).convert('RGB')
                img_t = transform(img_t).to(device)
                image_list.append(img_t)
            img_t = torch.stack(image_list, 0)
            fc_feature, att_feature = encoder(img_t)
            seq, probs, weights = decoder.decode_search(
                fc_feature, att_feature, beam_size
            )
            idx_range = range(idx, idx_end)
            sentecnes = decode_sequence(vocab, seq)
            for sent_idx, image_idx in enumerate(idx_range):
                entry = {'image_id': images[image_idx]['id'],
                         'caption': sentecnes[sent_idx]}
                descriptions.append(entry)
            logger.info("processed {}/{} images".format(
                idx_end, num_imgs
            ))
    logger.info("saving results file: {}".format(output_json))
    with open(output_json, 'w') as f:
        json.dump(descriptions, f)
    logger.info("Done")


if __name__ == '__main__':
    main()