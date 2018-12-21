from collections import OrderedDict
import time
import logging

import torch
import numpy as np

from image_captioning.utils.miscellaneous import decode_sequence
import sys
sys.path.append('coco-caption')
from pycocoevalcap.cider.cider import Cider

ciderd_scorer = Cider()


def get_self_critical_reward(
        sample_results,
        greed_results,
        all_captions,
        se_per_img,
        vocab
):
    logger = logging.getLogger('image_captioning.rewards')
    batch_size = sample_results.size(0)
    seq_length = sample_results.size(1)
    res = OrderedDict()

    sample_results = sample_results.cpu().detach().numpy()
    greed_results = greed_results.cpu().detach().numpy()
    sample_results = decode_sequence(vocab, sample_results)
    greed_results = decode_sequence(vocab, greed_results)
    for i in range(batch_size):
        res[i] = [sample_results[i]]
    for i in range(batch_size):
        res[batch_size+i] = [greed_results[i]]
    #res_ciderd = [{'image_id': i, 'caption': res[i]} for i in range(2*batch_size)]
    gts_str = [decode_sequence(vocab, gt) for gt in all_captions]
    gts = {i: gts_str[i%batch_size] for i in range(2*batch_size)}
    cider_scores_avg, cider_scores = ciderd_scorer.compute_score(gts, res)
    logger.info('average cider score: {:.4f}'.format(cider_scores_avg))
    scores = cider_scores[:batch_size] - cider_scores[batch_size:]
    rewards = np.repeat(scores[:, np.newaxis], seq_length, 1)
    return torch.from_numpy(rewards).to(torch.float32)




