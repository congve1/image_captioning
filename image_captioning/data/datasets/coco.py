import pickle
import json

import torch
import h5py


class COCODataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        att_features_file,
        fc_features_file,
        encoded_captions_file,
        encoded_captions_lens_file,
        seq_per_img
    ):
        self.root = root
        self.seq_per_img = seq_per_img
        att_features_file = h5py.File(att_features_file, 'r')
        fc_features_file = h5py.File(fc_features_file, 'r')
        self.att_features_dataset = att_features_file['att_features']
        self.fc_features_dataset = fc_features_file['fc_features']
        self.cocoid_dataset = fc_features_file['cocoids']
        with open(encoded_captions_file, 'r') as f:
            self.encoded_captions_file = json.load(f)
        with open(encoded_captions_lens_file, 'r') as f:
            self.encoded_captions_lens_flie = json.load(f)

    def __getitem__(self, index):
        att_feature = torch.from_numpy(
            self.att_features_dataset[index//self.seq_per_img]
        )
        fc_feature = torch.from_numpy(
            self.fc_features_dataset[index//self.seq_per_img]
        )
        cocoid = int(self.cocoid_dataset[index//self.seq_per_img])
        caption = torch.tensor(self.encoded_captions_file[index], dtype=torch.long)
        cap_len = torch.tensor(self.encoded_captions_lens_flie[index], dtype=torch.long)
        all_captions = torch.tensor(
            self.encoded_captions_file[(index//self.seq_per_img)*self.seq_per_img:
                                       ((index//self.seq_per_img)+1)*self.seq_per_img]
        )
        data = dict()
        data['att_feature'] = att_feature
        data['fc_feature'] = fc_feature
        data['cap_len'] = cap_len
        data['caption'] = caption
        data['all_captions'] = all_captions
        data['cocoid'] = cocoid
        return data

    def __len__(self):
        return len(self.encoded_captions_lens_flie)

