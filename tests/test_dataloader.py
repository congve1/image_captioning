import unittest

import torch

from image_captioning.data import make_data_loader
from image_captioning.config import cfg


class TestDataLoader(unittest.TestCase):
    def test_data_loader(self):
        cfg.merge_from_list(['DATASET.TRAIN', 'coco_2014_train_simple',
                             'DATASET.VAL', 'coco_2014_val_simple',
                             'DATASET.TEST', 'coco_2014_test_simple'])
        train_loader = make_data_loader(cfg)
        val_loader = make_data_loader(cfg, split='val')
        test_loader = make_data_loader(cfg, split='test')
        for loader in [train_loader, val_loader, test_loader]:
            if loader is train_loader:
                batch_size = cfg.SOLVER.IMS_PER_BATCH
            else:
                batch_size = cfg.TEST.IMS_PER_BATCH
            for batch_data in loader:
                self.assertEqual(
                    batch_data['att_features'].size(),
                    torch.Size([batch_size, 2048, cfg.MODEL.ENCODER.ATT_SIZE,
                                cfg.MODEL.ENCODER.ATT_SIZE])
                )
                self.assertEqual(
                    batch_data['fc_features'].size(),
                    torch.Size([batch_size, 2048])
                )
                self.assertEqual(
                    batch_data['captions'].size(),
                    torch.Size([batch_size, cfg.DATASET.SEQ_MAX_LEN+2])
                )
                self.assertEqual(
                    batch_data['all_captions'].size(),
                    torch.Size([batch_size, cfg.DATASET.SEQ_PER_IMG, cfg.DATASET.SEQ_MAX_LEN+2])
                )

    def test_data_loader_with_start_iter(self):
        cfg.merge_from_list(['DATASET.TRAIN', 'coco_2014_train_simple',
                             'DATASET.VAL', 'coco_2014_val_simple',
                             'DATASET.TEST', 'coco_2014_test_simple'])
        train_loader = make_data_loader(cfg, start_iter=1)
        val_loader = make_data_loader(cfg, split='val', start_iter=1)
        test_loader = make_data_loader(cfg, split='test', start_iter=1)
        for loader in [train_loader, val_loader, test_loader]:
            if loader is train_loader:
                batch_size = cfg.SOLVER.IMS_PER_BATCH
            else:
                batch_size = cfg.TEST.IMS_PER_BATCH
            for batch_data in loader:
                self.assertEqual(
                    batch_data['att_features'].size(),
                    torch.Size([batch_size, 2048, cfg.MODEL.ENCODER.ATT_SIZE,
                                cfg.MODEL.ENCODER.ATT_SIZE])
                )
                self.assertEqual(
                    batch_data['fc_features'].size(),
                    torch.Size([batch_size, 2048])
                )
                self.assertEqual(
                    batch_data['captions'].size(),
                    torch.Size([batch_size, cfg.DATASET.SEQ_MAX_LEN+2])
                )
                self.assertEqual(
                    batch_data['all_captions'].size(),
                    torch.Size([batch_size, cfg.DATASET.SEQ_PER_IMG, cfg.DATASET.SEQ_MAX_LEN+2])
                )

if __name__ == '__main__':
    unittest.main(verbosity=2)

