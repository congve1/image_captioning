import unittest

import torch

from image_captioning.config import cfg
from image_captioning.modeling.encoder import build_encoder
from image_captioning.utils.imports import import_file
from image_captioning.utils.checkpoint import ModelCheckpointer


class TestResNet(unittest.TestCase):
    def test_create_model(self):
        for structure_name in ['R-50-C4', 'R-50-C5', 'R-101-C4', 'R-101-C5']:
            cfg.merge_from_list(['MODEL.ENCODER.CONV_BODY', structure_name])
            model = build_encoder(cfg)
            dummy_input = torch.randn(1, 3, 224, 224)
            fc, att = model(dummy_input)
            relative_factor = int(structure_name[-1])   
            out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS * (2**relative_factor)
            att_size = cfg.MODEL.ENCODER.ATT_SIZE
            self.assertEqual(fc.size(), torch.Size((1,out_channels)))
            self.assertEqual(att.size(), torch.Size((1,out_channels, 
                                                     att_size, att_size)))

    def test_load_model(self):
        paths_catalog = import_file(
            'image_captioning.config.paths_catalog', cfg.PATHS_CATALOG, True
        )
        ResNetCatalog = paths_catalog.ResNetCatalog
        for structure_name in ['R-50-C4', 'R-50-C5', 'R-101-C4', 'R-101-C5']:
            cfg.merge_from_list(['MODEL.ENCODER.CONV_BODY', structure_name])
            model = build_encoder(cfg)
            dummy_input = torch.randn(1,3, 224,224)
            fc_dummy, att_dummy = model(dummy_input)
            model_checkpointer = ModelCheckpointer(cfg, model)
            model_checkpointer.load(ResNetCatalog.get(structure_name))
            fc_pre_trained, att_pre_trained = model(dummy_input)
            self.assertFalse((torch.abs(fc_dummy - fc_pre_trained)
                              < 1e-10).all())
            self.assertFalse((torch.abs(att_dummy- att_pre_trained) 
                              < 1e-10).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
