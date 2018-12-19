import torch
import torch.nn as nn
from torch.nn import functional as F
from image_captioning.modeling.utils import cat
from image_captioning.modeling import registry
from image_captioning.modeling.decoder.build import build_decoder_attention


@registry.DECODER_CORES.register("TopDownCore")
class TopDownCore(nn.Module):
    def __init__(self, cfg, vocab):
        super(TopDownCore, self).__init__()
        self.dropout_prob = cfg.MODEL.DECODER.DROPOUT_PROB
        self.vocab = vocab
        embedding_size = cfg.MODEL.DECODER.EMBEDDING_SIZE
        hidden_size = cfg.MODEL.DECODER.HIDDEN_SIZE
        self.att_lstm = nn.LSTMCell(
            embedding_size + hidden_size * 2, hidden_size
        )
        self.lang_lstm = nn.LSTMCell(
            hidden_size * 2, hidden_size
        )
        self.logit = nn.Linear(hidden_size, len(vocab))
        self.attention = build_decoder_attention(cfg)
        self._init_weights()

    def forward(self, xt, fc_feats, att_feats, hidden_states):
        prev_h_lang = hidden_states[0][1]
        prec_c_lang = hidden_states[1][1]
        prev_h_att = hidden_states[0][0]
        prev_c_att = hidden_states[1][0]
        input_att_lstm = cat([prev_h_lang, fc_feats, xt], 1)
        next_h_att, next_c_att= self.att_lstm(
            input_att_lstm, (prev_h_att, prev_c_att)
        )
        weighted_att_features, att_weights = self.attention(att_feats, next_h_att)
        input_lang_lstm = cat([weighted_att_features, next_h_att], 1)
        next_h_lang, next_c_lang = self.lang_lstm(
            input_lang_lstm, (prev_h_lang, prec_c_lang)
        )
        output = F.dropout(next_h_lang, self.dropout_prob, self.training)
        output = self.logit(output)
        hidden_states = (
            torch.stack([next_h_att, next_h_lang]),
            torch.stack([next_c_att, next_c_lang])
        )
        return output, hidden_states, att_weights

    def _init_weights(self):
        """
        use orthogonal to initialize lstm weights
        Returns:

        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)