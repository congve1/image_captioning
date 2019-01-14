import pickle
import json

import torch
import lmdb
import numpy as np
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


class COCODatasetLMDB(torch.utils.data.dataset.Dataset):
    def __init__(
        self,
        root,
        att_features_lmdb,
        fc_features_lmdb,
        encoded_captions_file,
        encoded_captions_lens_file,
        cocoids_file,
        seq_per_img,
        att_feature_shape,
        fc_feature_shape,
    ):
        self.root = root
        self.seq_per_img = seq_per_img
        self.att_feature_shape = att_feature_shape
        self.fc_feature_shape = fc_feature_shape
        with open(cocoids_file, 'r') as f:
            self.cocoids = json.load(f)
        self.encoded_captions = torch.load(
            encoded_captions_file,
            map_location='cpu'
        )
        self.encoded_captions_lens = torch.load(
            encoded_captions_lens_file,
            map_location='cpu'
        )
        self.att_features_lmdb = lmdb.open(
            att_features_lmdb, readonly=True, max_readers=1,
            lock=False, readahead=False, meminit=False
        )
        self.fc_features_lmdb = lmdb.open(
            fc_features_lmdb, readonly=True, max_readers=1,
            lock=False, readahead=False, meminit=False
        )
        with self.att_features_lmdb.begin(write=False) as txn:
            self.att_features_lmdb_keys = [key for key, _ in txn.cursor()]
        with self.fc_features_lmdb.begin(write=False) as txn:
            self.fc_features_lmdb_keys = [key for key, _ in txn.cursor()]

    def __getitem__(self, index):
        att_features_lmdb = self.att_features_lmdb
        fc_features_lmdb = self.fc_features_lmdb
        with att_features_lmdb.begin(write=False) as txn:
            att_feature = txn.get(
                self.att_features_lmdb_keys[index//self.seq_per_img]
            )
            att_feature = np.frombuffer(att_feature, dtype=np.float32)
            att_feature = att_feature.reshape(self.att_feature_shape)
            att_feature = torch.from_numpy(att_feature)
        with fc_features_lmdb.begin(write=False) as txn:
            fc_feature = txn.get(
                self.fc_features_lmdb_keys[index//self.seq_per_img]
            )
            fc_feature = np.frombuffer(fc_feature, dtype=np.float32)
            fc_feature = fc_feature.reshape(self.fc_feature_shape)
            fc_feature = torch.from_numpy(fc_feature)

        caption = self.encoded_captions[index]
        caption_len = self.encoded_captions_lens[index]
        all_captions = self.encoded_captions[
            (index//self.seq_per_img)*self.seq_per_img:
            ((index//self.seq_per_img)+1)*self.seq_per_img
        ]
        cocoid = self.cocoids[index//self.seq_per_img]
        return att_feature, fc_feature, caption, caption_len, all_captions, cocoid

    def __len__(self):
        return len(self.encoded_captions_lens)

