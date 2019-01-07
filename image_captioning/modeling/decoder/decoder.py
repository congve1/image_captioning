import torch
import torch.nn as nn
from torch.nn import functional as F

from image_captioning.utils.constant import epsilon
from image_captioning.modeling.decoder.build import build_decoder_core


def build_decoder(cfg, vocab):
    return Decoder(cfg, vocab)


class Decoder(nn.Module):
    def __init__(self, cfg, vocab):
        super(Decoder, self).__init__()
        self.vocab = vocab
        self.device = cfg.MODEL.DEVICE
        self.num_layers = 2
        self.hidden_size = cfg.MODEL.DECODER.HIDDEN_SIZE
        self.dropout_prob = cfg.MODEL.DECODER.DROPOUT_PROB
        self.feature_size = cfg.MODEL.ENCODER.FEATURE_SIZE
        self.seq_length = cfg.DATASET.SEQ_MAX_LEN
        self.ss_prob = 0.0 # scheduled sampling

        embedding_size = cfg.MODEL.DECODER.EMBEDDING_SIZE
        self.word_embed = nn.Sequential(
            nn.Embedding(len(vocab), embedding_size, padding_idx=vocab['<pad>']),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_prob)
        )
        self.att_embed = nn.Sequential(
            nn.Linear(self.feature_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_prob)

        )
        self.fc_embed = nn.Sequential(
            nn.Linear(self.feature_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_prob)
        )
        self.core = build_decoder_core(cfg, vocab)

    def init_hiddens(self, batch_size):
        weight = next(self.parameters())
        h_init = torch.zeros(
            (self.num_layers, batch_size, self.hidden_size),
            dtype=weight.dtype
        ).to(self.device)
        c_init = torch.zeros(
            (self.num_layers, batch_size, self.hidden_size),
            dtype=weight.dtype
        ).to(self.device)
        return h_init, c_init

    def forward(self, fc_feats, att_feats, seq):
        """

        Args:
            fc_feats: image global features
            att_feats: image spatial features
            seq: ground truth sequence

        Returns:
            outputs (torch.LongTensor): size(batch_size, seq_lenght-1, vocab_size),
                used to compute cross-entropy loss
            weights (torch.FloatTensor): size(batch_size, seq_length-1, spatial_locations),
                attention weights for generating each words
        """
        # put the feature dimension to the last
        batch_size = att_feats.size(0)
        att_feats = att_feats.permute(0, 2, 3, 1)
        locations = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att_feats = self.att_embed(att_feats)
        fc_feats = self.fc_embed(fc_feats)
        hidden_states = self.init_hiddens(batch_size)
        outputs = fc_feats.new_zeros(batch_size, seq.size(1)-1, len(self.vocab))
        weights = fc_feats.new_zeros(batch_size, seq.size(1)-1, locations)
        for i in range(seq.size(1)-1):
            if self.training and self.ss_prob > 0.0 and i >= 1:
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() <= epsilon:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].clone()
                    prob_prev = torch.exp(
                        self.get_log_probs(outputs[:, -2]).detach()
                    )
                    it.index_copy_(
                        0, sample_ind,
                        torch.multinomial(
                            prob_prev, 1
                        ).view(-1).index_select(0, sample_ind)
                    )
            else:
                it = seq[:, i].clone()
            if i >= 1 and self.is_all_sequences_end(it):
                break
            xt = self.word_embed(it)
            output, hidden_states, att_weights = self.core(
               xt, fc_feats, att_feats, hidden_states
            )
            outputs[:, i] = output
            weights[:, i] = att_weights
        return outputs, weights

    def decode_search(self, fc_feats, att_feats, beam_size=1):
        """
        if beam size > 1, use beam search , otherwise use greedy search
        Args:
            fc_feats:
            att_feats:
            beam_size:

        Returns:

        """
        if beam_size > 1:
            return self.beam_search(fc_feats, att_feats, beam_size)
        else:
            return self.greedy_search(fc_feats, att_feats)

    def beam_search(self, fc_feats, att_feats, beam_size):
        assert len(self.vocab) > beam_size, "assume beam size is smaller than vocab size"
        batch_size = fc_feats.size(0)
        att_feats = att_feats.permute(0, 2, 3, 1)
        locations = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        weights = fc_feats.new_zeros(batch_size, self.seq_length+1, locations)
        fc_feats = self.fc_embed(fc_feats)
        att_feats = self.att_embed(att_feats)
        seq = fc_feats.new_zeros((batch_size, self.seq_length+1), dtype=torch.long)\
            .fill_(self.vocab['<pad>'])
        seq_log_probs = fc_feats.new_zeros(batch_size, self.seq_length+1)
        # process every image independently for simplicity
        done_beams = [[] for _ in range(batch_size)]

        for k in range(batch_size):
            hidden_states = self.init_hiddens(beam_size)
            tmp_fc_feats = fc_feats[k:k+1].expand(beam_size, fc_feats.size(1))
            tmp_att_feats = att_feats[k:k+1].expand(
                *(beam_size,)+att_feats.size()[1:]
            ).contiguous()
            done_beams[k] = self._beam_search(
                tmp_fc_feats, tmp_att_feats, beam_size
            )
            seq[k, :] = done_beams[k][0]['seq'].to(self.device)# the first beam has highes cumulative score
            seq_log_probs[k, :] = done_beams[k][0]['logps'].to(self.device)
            weights[k, :] = done_beams[k][0]['att_weights'].to(self.device)
        return seq, seq_log_probs, weights

    def _beam_search(self, fc_feats, att_feats, beam_size):
        """
        perform beam_search on  one image
        Args:
            fc_feats:
            att_feats:

        Returns:

        """

        def beam_step(
                log_probs_f, beam_size,
                t, beam_seq, beam_seq_log_probs,
                beam_seq_log_porbs_sum, hidden_states, att_weights
        ):
            """
            complete one beam step
            Args:
                log_probs_f: log_probs on cpu
                beam_size: obvious
                t: time step
                beam_seq: tensor containing beams
                beam_seq_log_probs: tensor containing the beam logprobs
                beam_seq_log_porbs_sum: tensor containing joint logropbs
                hidden_states:

            Returns:
                beam_seq: tensor containing the word indices of the decoded captions
                beam_seq_log_probs : log-probability of each decision made,
                    same size of beam_seq
                beam_seq_log_probs_sum:
                new_hidden_states: new hidden_states
                candidates:
            """
            ys, ix = torch.sort(log_probs, 1, True)
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            candidates = []
            for r in range(rows):
                for c in range(cols):
                    local_log_prob = ys[r, c]
                    candiate_log_prob = beam_log_probs_sum[r] + local_log_prob
                    candidates.append(dict(
                        c=ix[r, c], r=r,
                        p=candiate_log_prob,
                        q=local_log_prob
                    ))
            candidates = sorted(candidates, key=lambda x: -x['p'])
            new_hidden_states = [_.clone() for _ in hidden_states]
            new_att_weights = att_weights.clone()
            if t >= 1:
                beam_seq_prev = beam_seq[:, :t].clone()
                beam_seq_log_probs_prev = beam_seq_log_probs[:, :t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                # fork beam index r into index vix
                if t >= 1:
                    beam_seq[vix, :t] = beam_seq_prev[v['r'], :]
                    beam_seq_log_probs[vix, :t] = beam_seq_log_probs_prev[v['r'], :]
                # rearrange hidden states
                for state_ix in range(len(new_hidden_states)):
                    new_hidden_states[state_ix][:, vix] = \
                    hidden_states[state_ix][:, v['r']]
                # rearrange att weights
                new_att_weights[vix] = att_weights[v['r']]
                # append new and terminal at the end of this beam
                beam_seq[vix, t] = v['c']  # c'th word is the continuation
                beam_seq_log_probs[vix, t] = v['q']  # the raw logprob here
                beam_log_probs_sum[vix] = v['p']
            return beam_seq, beam_seq_log_probs, beam_seq_log_porbs_sum,\
                new_hidden_states, new_att_weights, candidates
        hidden_states = self.init_hiddens(beam_size)
        locations = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        beam_seq = torch.zeros((beam_size, self.seq_length+1), dtype=torch.long)
        beam_seq_log_probs = torch.zeros((beam_size, self.seq_length+1))
        beam_log_probs_sum = torch.zeros(beam_size)
        beam_att_weights = att_feats.new_zeros(
            beam_size,
            self.seq_length+1,
            locations,
        )
        done_beams = []
        for t in range(self.seq_length+1):
            """
            pem a beam merge. that is,
            for every previous beam we now many new possibilities to branch out
            we need to resort our beams to maintain the loop invariant for keeping
            the top beam_size most likely sequences.
            """
            if t == 0:
                it = fc_feats.new_zeros([beam_size], dtype=torch.long).fill_(self.vocab['<start>'])
            xt = self.word_embed(it.to(self.device))
            outputs, hidden_states, att_weights = self.core(
                xt, fc_feats,
                att_feats, hidden_states
            )
            log_probs = self.get_log_probs(outputs)
            # lets go to cpu for more efficiency in indexing operations
            log_probs = log_probs.cpu().float()
            # supress UNK tokens in the decoding
            unk_idx = self.vocab['<unk>']
            log_probs[:, unk_idx] = log_probs[:, unk_idx] - 1000
            beam_seq,\
            beam_seq_log_probs,\
            beam_log_probs_sum,\
            hidden_states,\
            att_weights,\
            candidates_divm = beam_step(
                log_probs,
                beam_size,
                t,
                beam_seq,
                beam_seq_log_probs,
                beam_log_probs_sum,
                hidden_states,
                att_weights
            )
            beam_att_weights[:, t] = att_weights
            for vix in range(beam_size):
                # if time'up.. or if end token is reached then copy beams
                if beam_seq[vix, t] == self.vocab['<end>'] or t == self.seq_length:
                    final_beam = {
                        'seq': beam_seq[vix, :].clone(),
                        'logps': beam_seq_log_probs[vix, :].clone(),
                        'p': beam_log_probs_sum[vix].clone(),
                        'att_weights': beam_att_weights[vix].clone(),
                    }
                    done_beams.append(final_beam)
                    # don't continue beams from finished sequences
                    beam_log_probs_sum[vix] = -1000
            # encode as vectors
            it = beam_seq[:, t]
        done_beams = sorted(
            done_beams, key=lambda x: -x['p']
        )[:beam_size]
        return done_beams

    def greedy_search(self, fc_feats, att_feats):
        batch_size = fc_feats.size(0)
        att_feats = att_feats.permute(0, 2, 3, 1)
        locations = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        weights = fc_feats.new_zeros(batch_size, self.seq_length+1, locations)
        fc_feats = self.fc_embed(fc_feats)
        att_feats = self.att_embed(att_feats)
        hidden_states = self.init_hiddens(batch_size)
        seq = fc_feats.new_zeros((batch_size, self.seq_length+1), dtype=torch.long)\
            .fill_(self.vocab['<pad>'])
        seq_log_probs = fc_feats.new_zeros(batch_size, self.seq_length+1)
        for t in range(self.seq_length+1):
            if t == 0:  # input <start>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long).fill_(self.vocab['<start>'])
            xt = self.word_embed(it)
            output, hidden_states, att_weights = self.core(xt, fc_feats, att_feats, hidden_states)
            log_probs = self.get_log_probs(output)
            sample_log_probs, idxs = torch.max(log_probs, 1)
            it = idxs.view(-1).long()
            seq[:, t] = it
            seq_log_probs[:, t] = sample_log_probs.view(-1)
            weights[:, t] = att_weights
            if self.is_all_sequences_end(it):
                break
        return seq, seq_log_probs, weights

    def sample(self, fc_feats, att_feats):
        batch_size = fc_feats.size(0)
        att_feats = att_feats.permute(0, 2, 3, 1)
        locations = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        weights = fc_feats.new_zeros(batch_size, self.seq_length + 1, locations)
        fc_feats = self.fc_embed(fc_feats)
        att_feats = self.att_embed(att_feats)
        hidden_states = self.init_hiddens(batch_size)
        seq = fc_feats.new_zeros((batch_size, self.seq_length + 1), dtype=torch.long) \
            .fill_(self.vocab['<pad>'])
        seq_log_probs = fc_feats.new_zeros(batch_size, self.seq_length + 1)

        for t in range(self.seq_length + 1):
            if t == 0:  # input <start>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long).fill_(self.vocab['<start>'])
            xt = self.word_embed(it)
            output, hidden_states, att_weights = self.core(xt, fc_feats, att_feats, hidden_states)
            log_probs = self.get_log_probs(output)
            probs = torch.exp(log_probs)
            it = torch.multinomial(probs, 1)
            sample_log_probs = log_probs.gather(1, it)
            it = it.view(-1).long()
            seq[:, t] = it
            seq_log_probs[:, t] = sample_log_probs.view(-1)
            weights[:, t] = att_weights
            if self.is_all_sequences_end(it):
                break
        return seq, seq_log_probs, weights

    def get_log_probs(self, logits):
        probs = F.log_softmax(logits, dim=1)
        return probs

    def is_all_sequences_end(self, it):
        """
        Check whether all teh sequences are end.
        Args:
            it (torch.LongTensor): the words of one time step.

        Returns:
            bool : True if all sequence ends. Otherwise else.
        """
        end_idx = self.vocab['<end>']
        pad_idx = self.vocab['<pad>']
        num_end_words = (it == end_idx).sum()
        num_pad_words = (it == pad_idx).sum()
        if it.sum() == num_end_words*end_idx + num_pad_words*pad_idx:
            return True
        return False
