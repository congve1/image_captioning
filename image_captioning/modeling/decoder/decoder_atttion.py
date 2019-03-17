import torch
import torch.nn as nn
import torch.nn.functional as F
from image_captioning.modeling import registry


def build_decoder_attention(cfg, att=None):
    att_name = cfg.MODEL.DECODER.ATTENTION if att is None else att
    assert att_name in registry.DECODER_ATTENTIONS, \
        "DECODER.ATTENTION: {} is not registered in registry".format(
            att_name
        )
    return registry.DECODER_ATTENTIONS[att_name](cfg)


@registry.DECODER_ATTENTIONS.register("TopDownAttention")
class TopDownAttention(nn.Module):
    def __init__(self, cfg):
        super(TopDownAttention, self).__init__()
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
        weights = weights.squeeze(2)
        weights = F.softmax(weights, dim=1)
        att_features = att_features.view(-1, locations, att_features.size(-1))
        weighted_att_features = torch.bmm(
            weights.unsqueeze(1), att_features
        )
        return weighted_att_features.squeeze(1), weights


@registry.DECODER_ATTENTIONS.register('DualAttention')
class DualAttention(nn.Module):
    def __init__(self, cfg):
        super(DualAttention, self).__init__()
        att_hid_size = cfg.MODEL.DECODER.ATT_HIDDEN_SIZE
        hidden_size = cfg.MODEL.DECODER.HIDDEN_SIZE
        att_size = cfg.MODEL.ENCODER.ATT_SIZE

        self.att2att = nn.Linear(hidden_size, att_hid_size)
        self.h2att = nn.Linear(hidden_size, att_hid_size)
        self.alpha_net = nn.Linear(att_hid_size, 1)

        self.channel_to_att = nn.Linear(att_size*att_size, att_hid_size)
        self.beta_net = nn.Linear(att_hid_size, 1)

    def forward(self, att_features, h):
        # att_features: BxHxWxC
        locations = att_features.numel()//att_features.size(0)//att_features.size(-1)
        # size: BxLxC
        att_features = att_features.reshape(-1, locations, att_features.size(-1))
        spatial_att_features = self.att2att(att_features)
        spatial_att_features = spatial_att_features.view(
            -1, locations, spatial_att_features.size(-1)
        )
        channel_att_features = self.channel_to_att(att_features.permute(0, 2, 1))
        p_h = self.h2att(h)
        # match the dimension with p_att_features
        p_h = p_h.unsqueeze(1)
        # size: BxL
        weights_spatial = self.alpha_net(
            torch.tanh(spatial_att_features + p_h)
        ).squeeze(2)
        weights_spatial = F.softmax(weights_spatial, dim=1)
        # size: BxC
        weighted_spatial_features = torch.bmm(
            weights_spatial.unsqueeze(1), att_features
        ).squeeze(1)
        # size: BxC
        weights_channel = self.beta_net(
            torch.tanh(channel_att_features+p_h)
        ).squeeze(2)
        weights_channel = F.softmax(weights_channel, dim=1)
        # size: BxL
        weighted_channel_features = torch.bmm(
            weights_channel.unsqueeze(1), att_features.permute(0, 2, 1)
        ).squeeze(1)
        return weighted_spatial_features,\
            weighted_channel_features,\
            weights_spatial


@registry.DECODER_ATTENTIONS.register("ChannelAttention")
class ChannelAttention(nn.Module):
    def __init__(self, cfg):
        super(ChannelAttention, self).__init__()
        att_hid_size = cfg.MODEL.DECODER.ATT_HIDDEN_SIZE
        hidden_size = cfg.MODEL.DECODER.HIDDEN_SIZE
        att_size = cfg.MODEL.ENCODER.ATT_SIZE

        self.channel2att = nn.Linear(att_size*att_size, att_hid_size)
        self.h2att = nn.Linear(hidden_size, att_hid_size)
        self.alpha_net = nn.Linear(att_hid_size, 1)

    def forward(self, att_features, h):
        # att_features: BxHxWxC
        locations = att_features.numel()//att_features.size(0)//att_features.size(-1)
        att_features = att_features.reshape((att_features.size(0), locations,
                                             att_features.size(-1))).permute(0, 2, 1)
        channel_att_features = self.channel2att(att_features)
        p_h = self.h2att(h)
        # match the dimension with p_att_features
        p_h = p_h.unsqueeze(1)
        weights = self.alpha_net(
            torch.tanh(channel_att_features + p_h)
        )
        weights = weights.squeeze(2)
        weights = F.softmax(weights, dim=1)
        weighted_att_features = torch.bmm(
            weights.unsqueeze(1), att_features
        )
        return weighted_att_features.squeeze(1), weights
