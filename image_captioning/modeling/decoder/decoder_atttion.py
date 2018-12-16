import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, cfg):
        super(Attention, self).__init__()
        att_hid_size = cfg.MODEL.DECODER.ATT_HIDDEN_SIZE
        hidden_size = cfg.MODEL.DECODER.HIDDEN_SIZE

        self.att2att = nn.Linear(hidden_size, att_hid_size)
        self.h2att = nn.Linear(hidden_size, att_hid_size)
        self.alpha_net = nn.Linear(att_hid_size, 1)

    def forward(self, att_features, h):
        # att_features: BxHxWxC
        locations = att_features.numel()//att_features.size(0)//att_features.size(-1)
        p_att_features = self.att2att(att_features)
        p_att_features = p_att_features.view(
            -1, locations, p_att_features.size(-1)
        )
        p_h = self.h2att(h)
        # match the dimension with p_att_features
        p_h = p_h.unsqueeze(1)
        weights = self.alpha_net(
            torch.tanh(p_att_features + p_h)
        )
        weights = weights.squeeze()
        weights = F.softmax(weights, dim=1)
        att_features = att_features.view(-1, locations, att_features.size(-1))
        weighted_att_features = torch.bmm(
            weights.unsqueeze(1), att_features
        )
        return weighted_att_features.squeeze(1), weights
