from collections import OrderedDict
import os
from tempfile import TemporaryDirectory
import unittest

import torch
from torch import nn

from image_captioning.utils.model_serialization import load_state_dict
from image_captioning.utils.checkpoint import Checkpointer, ModelCheckpointer
from image_captioning.config import cfg
from image_captioning.modeling.encoder import build_encoder
from image_captioning.utils.model_zoo import cache_url


class TestCheckpointer(unittest.TestCase):
    def create_model(self):
        return nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
    
    def create_complex_model(self):
        m = nn.Module()
        m.block1 = nn.Module()
        m.block1.layer1 = nn.Linear(2, 3)
        m.layer2 = nn.Linear(3, 2)
        m.res = nn.Module()
        m.res.layer2 = nn.Linear(3, 2)

        state_dict = OrderedDict()
        state_dict['layer1.weight'] = torch.randn(3, 2)
        state_dict['layer1.bias'] = torch.randn(3)
        state_dict['layer2.weight'] = torch.randn(2, 3)
        state_dict['layer2.bias'] = torch.randn(2)
        state_dict['res.layer2.weight'] = torch.rand(2, 3)
        state_dict['res.layer2.bias'] = torch.rand(2)

        return m, state_dict
    def test_from_last_checkpoint_model(self):
        # test that loading works even if they differ by a prefix
        for trained_model, fresh_model in [
            (self.create_model(), self.create_model()),
            (nn.DataParallel(self.create_model()), self.create_model()),
            (self.create_model(), nn.DataParallel(self.create_model())),
            (
                nn.DataParallel(self.create_model()),
                nn.DataParallel(self.create_model())
            ),
        ]:
            with TemporaryDirectory() as f:
                checkpointer = Checkpointer(
                    trained_model, save_dir=f, save_to_disk=True
                )
                checkpointer.save('checkpoint_file')

                # in the same folder
                fresh_checkpointer = Checkpointer(fresh_model, save_dir=f)
                self.assertTrue(fresh_checkpointer.has_checkpoint())
                self.assertEqual(
                    fresh_checkpointer.get_checkpoint_file(),
                    os.path.join(f, "checkpoint_file.pth")
                )
                _ = fresh_checkpointer.load()
                for trained_p, loaded_p in zip(
                    trained_model.parameters(), fresh_model.parameters()
                ):
                    # different tensor references
                    self.assertFalse(id(trained_p) == id(loaded_p))
                    # same content
                    self.assertTrue(trained_p.equal(loaded_p))
    def test_from_name_file_model(self):
        # test that loading works even if they differ by a prefix
        for trained_model, fresh_model in [
            (self.create_model(), self.create_model()),
            (nn.DataParallel(self.create_model()), self.create_model()),
            (self.create_model(), nn.DataParallel(self.create_model())),
            (
                nn.DataParallel(self.create_model()),
                nn.DataParallel(self.create_model())
            ),
        ]:
            with TemporaryDirectory() as f:
                checkpointer = Checkpointer(
                    trained_model, save_dir=f, save_to_disk=True
                )
                checkpointer.save('checkpoint_file')
            # on different folders
                with TemporaryDirectory() as g:
                    fresh_checkpointer = Checkpointer(fresh_model, save_dir=g)
                    self.assertFalse(fresh_checkpointer.has_checkpoint())
                    self.assertEqual(fresh_checkpointer.get_checkpoint_file(), "")
                    _ = fresh_checkpointer.load(os.path.join(f, 'checkpoint_file.pth'))
            for trained_p, loaded_p in zip(
                    trained_model.parameters(), fresh_model.parameters()
                ):
                    # different tensor references
                    self.assertFalse(id(trained_p) == id(loaded_p))
                    # same content
                    self.assertTrue(trained_p.equal(loaded_p))
    def test_complex_model_loaded(self):
        for add_data_parallel in [False, True]:
            model, state_dict = self.create_complex_model()
            if add_data_parallel:
                model = nn.DataParallel(model)
            load_state_dict(model, state_dict)
            for loaded, stored in zip(model.state_dict().values(), state_dict.values()):
                #different tensor references
                self.assertFalse(id(loaded) == id(stored))
                # same content
                self.assertTrue(loaded.equal(stored))
class TestModelCheckpointer(unittest.TestCase):
    def test_load_from_catalog(self):
        pass
    def test_load_from_url(self):
        model_urls = {
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        }
        url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
        cfg.merge_from_list(['MODEL.ENCODER.CONV_BODY', 'R-101-C5'])
        model = build_encoder(cfg)
        model_checkpointer = ModelCheckpointer(cfg, model)
        model_checkpointer.load(url)

        cached_f = cache_url(url)
        cached_state_dict = torch.load(cached_f, map_location='cpu')




if __name__ == "__main__":
    unittest.main(verbosity=2)