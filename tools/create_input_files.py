"""
using h5py to create att_features file, fc_features_file and encoded_captions_file
"""
import json
import pickle
import logging
import argparse
from random import seed, choice, sample
import os

import PIL.Image as Image
import numpy as np
import torch

from image_captioning.utils.imports import import_file
from image_captioning.config import cfg
from image_captioning.utils.get_vocab import get_vocab
from image_captioning.utils.checkpoint import ModelCheckpointer
from image_captioning.modeling.encoder import build_encoder
from image_captioning.utils.logger import setup_logger
from image_captioning.data.transforms.build import build_transforms
from image_captioning.utils.miscellaneous import encode_caption
from image_captioning.utils.miscellaneous import mkdir


def create_input_files(args):
    logger = setup_logger('image_captioning')
    logger.info("merge options from list {}".format(args.opts))
    if args.config_file:
        cfg.merge_from_file(args.config_file)
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = '\n' + cf.read()
            logger.info(config_str)
    cfg.merge_from_list(args.opts)
    paths_catalog = import_file(
        'image_captioning.config.paths_catalog', cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    ResNetCatalog = paths_catalog.ResNetCatalog
    encoder = build_encoder(cfg)
    encoder.eval()
    weight_loader = ModelCheckpointer(cfg, encoder)
    url = ResNetCatalog.get(cfg.MODEL.ENCODER.CONV_BODY)
    weight_loader.load(url)
    device = cfg.MODEL.DEVICE
    encoder.to(device)
    seq_max_len = cfg.DATASET.SEQ_MAX_LEN
    seq_per_img = cfg.DATASET.SEQ_PER_IMG
    transform = build_transforms()
    att_size = cfg.MODEL.ENCODER.ATT_SIZE
    att_dim = 2048 if cfg.MODEL.ENCODER.CONV_BODY.endswith('5') else 1024
    for dataset, split in [(cfg.DATASET.TRAIN, 'train'),
                           (cfg.DATASET.VAL, 'val'),
                           (cfg.DATASET.TEST, 'test')]:
        if not dataset:
            continue
        data = DatasetCatalog.get(dataset)
        root = data['args']['root']
        ann_file = data['args']['ann_file']
        att_features_folder = os.path.abspath(os.path.join(root, dataset+"_att_features"))
        fc_features_folder = os.path.abspath(os.path.join(root, dataset+'_fc_features'))
        if not os.path.exists(att_features_folder):
            mkdir(att_features_folder)
        if not os.path.exists(fc_features_folder):
            mkdir(fc_features_folder)
        att_features_paths_file = data['args']['att_features_paths_file']
        fc_features_paths_file = data['args']['fc_features_paths_file']
        encoded_captions_file = data['args']['encoded_captions_file']
        encoded_captions_lens_file = data['args']['encoded_captions_lens_file']
        cocoids_file = data['args']['cocoids_file']
        vocab = get_vocab(dataset)
        logger.info("start processing dataset {}".format(dataset))
        img_paths = []
        captions = []
        coco_ids = []
        att_features_paths = []
        fc_features_paths = []
        logger.info("start loading annotation file {}".format(ann_file))
        with open(ann_file, 'r') as f:
            ann_file = json.load(f)
        logger.info("annotation file loaded")
        for idx, image in enumerate(ann_file['images']):
            # getting all the captions tokens associate with the image
            img_captions = []
            for sentence in image['sentences']:
                if len(sentence['tokens']) <= seq_max_len:
                    img_captions.append(sentence['tokens'])
                else:
                    img_captions.append(sentence['tokens'][:seq_max_len])
            path = os.path.abspath(os.path.join(root, 
                                                image['filepath'],
                                                image['filename']))
            if split == 'train' and image['split'] in ['train', 'restval']:
                img_paths.append(path)
                captions.append(img_captions)
                coco_ids.append(image['cocoid'])
            elif split == image['split']:
                img_paths.append(path)
                captions.append(img_captions)
                coco_ids.append(image['cocoid'])
            if (idx+1) % 1000 == 0:
                logger.info("assign {}/{} images".format(idx+1, len(ann_file['images'])))

        logger.info("assign {} imgs for split '{}' in dataset '{}'".format
                    (len(img_paths), split, dataset))
        logger.info("assign {} sentences for split '{}' in dataset '{}'".format
                    (sum([len(img_captions) for img_captions in captions]),
                        split, dataset))
        # write features into h5py file
        enc_cptions = []
        caplens = []
        seed(123)
        batch_size = cfg.TEST.IMS_PER_BATCH
        img_nums = len(img_paths)
        logger.info("start processing images and captions")
        # batched processing
        for idx in range(0, img_nums, batch_size):
            idx_end = idx + batch_size
            if idx_end > img_nums:
                idx_end = img_nums
            imgs = []
            image_captions = []
            for idy in range(idx, idx_end):
                # get imgs
                img = Image.open(img_paths[idy]).convert("RGB")
                img = transform(img)
                imgs.append(img)
                # get captions
                if (len(captions[idy])) < seq_per_img:
                    captions_one_img = (
                            captions[idy]
                            + [choice(captions[idy]) for _ in range(seq_per_img - len(captions[idy]))]
                    )
                else:
                    captions_one_img = sample(captions[idy], k=seq_per_img)
                image_captions.extend(captions_one_img)
            imgs = torch.stack(imgs).to(device)
            # get features
            with torch.no_grad():
                fc, att = encoder(imgs)
            idx_list = list(range(idx, idx_end))
            for feature_idx, path_idx in enumerate(idx_list):
                file_name = os.path.basename(img_paths[path_idx]).split('.')[0]
                att_feature_path = os.path.abspath(
                    os.path.join(att_features_folder, file_name+"_att_feature.pt")
                )
                fc_feature_path = os.path.abspath((
                   os.path.join(fc_features_folder, file_name+'_fc_feature.pt')
                ))
                att_feature = att[feature_idx].clone()
                fc_feature = fc[feature_idx].clone()
                torch.save(att_feature, att_feature_path)
                torch.save(fc_feature, fc_feature_path)
                att_features_paths.append(att_feature_path)
                fc_features_paths.append(fc_feature_path)
            # get encode captions and cap lens
            for i, cap in enumerate(image_captions):
                c_len = min(len(cap), seq_max_len)  # length without <start> <end> token
                cap = ['<start>'] + cap + ['<end>'] + \
                      ['<pad>' for _ in range(seq_max_len-c_len)]
                enc_cptions.append(encode_caption(vocab, cap))
                caplens.append(c_len)
            logger.info(
                "processed images: {}/{}. encoded captions. {}/{}"
                .format(
                    (idx_end), img_nums,
                    (idx_end)*seq_per_img, img_nums*seq_per_img
                )
            )
        enc_cptions = torch.tensor(enc_cptions, dtype=torch.long)
        caplens = torch.tensor(caplens, dtype=torch.long)
        logger.info("writing att feature paths to file {}"
                    .format(fc_features_paths_file))
        with open(att_features_paths_file, 'w') as f:
            json.dump(att_features_paths, f)

        logger.info("writing fc feature paths to file {}"
                    .format(att_features_paths_file))
        with open(fc_features_paths_file, 'w') as f:
            json.dump(fc_features_paths, f)
        logger.info("writing coocids to fie {}"
                    .format(cocoids_file))
        with open(cocoids_file, 'w') as f:
            json.dump(coco_ids, f)
        logger.info("writing encoded captions to file {}"
                    .format(encoded_captions_file))
        # Save encoded captions and their lengths to JSON files
        torch.save(enc_cptions, encoded_captions_file)
        logger.info("writing captions lengths to file {}"
                    .format(encoded_captions_lens_file))
        torch.save(caplens, encoded_captions_lens_file)


if __name__ == '__main__':
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
    create_input_files(args)

