import unittest

import torch

from image_captioning.config import cfg
from image_captioning.modeling.decoder.decoder_atttion import Attention
from image_captioning.modeling.decoder.decoder_core import DecoderCore
from image_captioning.modeling.decoder.decoder import Decoder
from image_captioning.utils.get_vocab import get_vocab


class TestDecoder(unittest.TestCase):
    batch_size = 8
    num_layers = 2
    feature_size = cfg.MODEL.ENCODER.FEATURE_SIZE
    att_size = cfg.MODEL.ENCODER.ATT_SIZE
    att_hid_size = cfg.MODEL.DECODER.ATT_HIDDEN_SIZE
    embedding_size = cfg.MODEL.DECODER.EMBEDDING_SIZE
    hidden_size = cfg.MODEL.DECODER.HIDDEN_SIZE
    seq_length = cfg.DATASET.SEQ_MAX_LEN + 2
    vocab = get_vocab('coco_2014')

    def get_dummy_fc_features(self):
        return torch.randn(
            TestDecoder.batch_size,
            TestDecoder.feature_size
        )

    def get_dummy_fc_features_p(self):
        return torch.randn(
            TestDecoder.batch_size,
            TestDecoder.hidden_size
        )

    def get_dummy_att_features(self):
        return torch.randn(
            TestDecoder.batch_size,
            TestDecoder.feature_size,
            TestDecoder.att_size,
            TestDecoder.att_size
        )

    def get_dummy_att_features_p(self):
        return torch.randn(
            TestDecoder.batch_size,
            TestDecoder.att_size,
            TestDecoder.att_size,
            TestDecoder.hidden_size
        )

    def get_dummy_hidden_states(self):
           return torch.randn(
               TestDecoder.num_layers, TestDecoder.batch_size, TestDecoder.hidden_size
           ),\
            torch.randn(
                TestDecoder.num_layers, TestDecoder.batch_size, TestDecoder.hidden_size
            )

    def get_dummy_h(self):
        return torch.randn(
            TestDecoder.batch_size,
            TestDecoder.hidden_size
        )

    def get_dummy_xt(self):
        return torch.randn(
            TestDecoder.batch_size,
            TestDecoder.embedding_size
        )

    def get_dummy_seq(self):
        return torch.randint(
            9000, (TestDecoder.batch_size, TestDecoder.seq_length),
            dtype=torch.long
        )

    def test_attention(self):
        attention = Attention(cfg)
        dummy_att_features = self.get_dummy_att_features_p()
        dummy_h = self.get_dummy_h()
        a, w = attention(dummy_att_features, dummy_h)
        self.assertEqual(
            a.size(),
            torch.Size([TestDecoder.batch_size,  dummy_att_features.size(-1)])
        )
        self.assertEqual(
            w.size(),
            torch.Size([TestDecoder.batch_size, TestDecoder.att_size*TestDecoder.att_size])
        )

    def test_core(self):
        dummy_att_features = self.get_dummy_att_features_p()
        dummy_fc_features = self.get_dummy_fc_features_p()
        dummy_hiddens = self.get_dummy_hidden_states()
        dummy_xt = self.get_dummy_xt()
        core = DecoderCore(cfg, TestDecoder.vocab)
        output, hidden_states, weights = core(
            dummy_xt, dummy_fc_features, dummy_att_features,
            dummy_hiddens
        )
        self.assertEqual(
            output.size(),
            torch.Size([TestDecoder.batch_size, len(TestDecoder.vocab)])
        )
        self.assertEqual(
            hidden_states[0].size(),
            dummy_hiddens[0].size()
        )
        self.assertEqual(
            hidden_states[1].size(),
            dummy_hiddens[1].size()
        )
        self.assertEqual(
            weights.size(),
            torch.Size([TestDecoder.batch_size, TestDecoder.att_size*TestDecoder.att_size])
        )

    def test_decoder(self):
        dummy_fc_features = self.get_dummy_fc_features()
        dummy_att_features = self.get_dummy_att_features()
        seq = self.get_dummy_seq()
        vocab = TestDecoder.vocab
        decoder = Decoder(cfg, vocab)
        outputs, weights = decoder(
            dummy_fc_features, dummy_att_features, seq
        )
        self.assertEqual(
            outputs.size(),
            torch.Size([
                TestDecoder.batch_size,
                TestDecoder.seq_length-1,
                len(vocab)
            ])
        )
        self.assertEqual(
            weights.size(),
            torch.Size([
                TestDecoder.batch_size,
                TestDecoder.seq_length-1,
                TestDecoder.att_size*TestDecoder.att_size
            ])
        )
        seq, seqprobs, weights = decoder.decode_search(dummy_fc_features,
                                                       dummy_att_features)
        self.assertEqual(
            seq.size(),
            torch.Size([TestDecoder.batch_size, TestDecoder.seq_length-1])
        )
        self.assertEqual(
            seqprobs.size(),
            torch.Size([TestDecoder.batch_size, TestDecoder.seq_length-1])
        )
        self.assertEqual(
            weights.size(),
            torch.Size([
                TestDecoder.batch_size,
                TestDecoder.seq_length-1,
                TestDecoder.att_size*TestDecoder.att_size
            ])
        )
        seq, seqprobs, weights = decoder.decode_search(dummy_fc_features,
                                                       dummy_att_features,
                                                       3)
        self.assertEqual(
            seq.size(),
            torch.Size([TestDecoder.batch_size, TestDecoder.seq_length-1])
        )
        self.assertEqual(
            seqprobs.size(),
            torch.Size([TestDecoder.batch_size, TestDecoder.seq_length-1])
        )
        self.assertEqual(
            weights.size(),
            torch.Size([
                TestDecoder.batch_size,
                TestDecoder.seq_length-1,
                TestDecoder.att_size*TestDecoder.att_size
            ])
        )
        seq, seqprobs, weights = decoder.sample(dummy_fc_features,
                                                dummy_att_features)
        self.assertEqual(
            seq.size(),
            torch.Size([TestDecoder.batch_size, TestDecoder.seq_length - 1])
        )
        self.assertEqual(
            seqprobs.size(),
            torch.Size([TestDecoder.batch_size, TestDecoder.seq_length - 1])
        )
        self.assertEqual(
            weights.size(),
            torch.Size([
                TestDecoder.batch_size,
                TestDecoder.seq_length - 1,
                TestDecoder.att_size * TestDecoder.att_size
            ])
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)