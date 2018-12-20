import pickle
import json

import torch
import h5py
import logging
import time


class COCODataset(torch.utils.data.dataset.Dataset):
    def __init__(
        self,
        root,
        att_features_paths_file,
        fc_features_paths_file,
        encoded_captions_file,
        encoded_captions_lens_file,
        cocoids_file,
        seq_per_img
    ):
        self.root = root
        self.seq_per_img = seq_per_img
        with open(att_features_paths_file, 'r') as f:
            self.att_features_paths = json.load(f)
        with open(fc_features_paths_file, 'r') as f:
            self.fc_features_paths = json.load(f)
        with open(cocoids_file, 'r') as f:
            self.cocoids = json.load(f)
        self.encoded_captions = torch.load(encoded_captions_file,
                                           map_location='cpu')
        self.encoded_captions_lens = torch.load(encoded_captions_lens_file,
                                                map_location='cpu')

    def __getitem__(self, index):

        att_feature = torch.load(
            self.att_features_paths[index//self.seq_per_img],
            map_location='cpu'
        )
        fc_feature = torch.load(
            self.fc_features_paths[index//self.seq_per_img],
            map_location='cpu'
        )
        cap_len = self.encoded_captions_lens[index]
        caption = self.encoded_captions[index]
        all_captions = self.encoded_captions[
            (index//self.seq_per_img)*self.seq_per_img:
            ((index//self.seq_per_img)+1)*self.seq_per_img
        ]
        cocoid = self.cocoids[index//self.seq_per_img]
        data = dict()
        data['att_feature'] = att_feature
        data['fc_feature'] = fc_feature
        data['cap_len'] = cap_len
        data['caption'] = caption
        data['all_captions'] = all_captions
        data['cocoid'] = cocoid
        return data

    def __len__(self):
        return len(self.encoded_captions_lens)


