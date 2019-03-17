import datetime
import logging
import time
import os

import torch
from tqdm import tqdm
from torch import distributed as dist

from image_captioning.utils.miscellaneous import decode_sequence
from image_captioning.data.datasets.evaluation import coco_eval
from image_captioning.utils.comm import get_world_size
from image_captioning.utils.comm import synchronize
from image_captioning.utils.comm import all_gather
from image_captioning.utils.comm import is_main_process


def compute_on_dataset(
        model, criterion, data_loader, vocab, beam_size, device, logger,
):
    model.eval()
    cpu_device = torch.device("cpu")
    val_loss_sum = 0.
    val_loss_count = 0
    seq_per_img = data_loader.dataset.seq_per_img
    predictions = []
    done_ids = dict()
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader, ncols=100, ascii=True, desc="decoding")):
            fc_features = data['fc_features'].to(device)
            att_fatures = data['att_features'].to(device)
            captions = data['captions'].to(device)
            cap_lens = data['cap_lens'].to(device)
            cocoids = data['cocoids']
            outputs, *_ = model(fc_features, att_fatures, captions)
            loss = criterion(outputs, captions[:, 1:], cap_lens+1)
            val_loss = loss.item()
            val_loss_count += 1
            val_loss_sum += val_loss
            seqs, seq_log_probs, *_ = model.decode_search(
                fc_features, att_fatures, beam_size=beam_size
            )
            sents = decode_sequence(vocab, seqs)
            for k, sent in enumerate(sents):
                entry = {'image_id': cocoids[k], 'caption': sent}
                if cocoids[k] not in done_ids:
                    done_ids[cocoids[k]] = 0
                    predictions.append(entry)
    return predictions, val_loss_sum / val_loss_count


def reduce_loss(loss):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size =get_world_size()
    if world_size < 2:
        return loss
    with torch.no_grad():
        dist.reduce(loss, dst=0)
        if dist.get_rank() == 0:
            loss /= world_size
    return loss


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    predictions = []
    image_ids = []
    for p in all_predictions:
        for entry in p:
            if entry['image_id'] not in image_ids:
                image_ids.append(entry['image_id'])
                predictions.append(entry)
    return predictions

def inference(
        model,
        criterion,
        data_loader,
        dataset_name,
        vocab,
        beam_size,
        device='cpu',
):
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("image_captioning.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images)".format(dataset_name, len(dataset)))
    start_time = time.time()
    predictions, loss = compute_on_dataset(
        model, criterion, data_loader, vocab, beam_size, device, logger,
    )
    loss_reduced = reduce_loss(torch.tensor(loss).to(device))
    synchronize()
    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return None, None, None
    metrics_score = coco_eval(predictions, dataset_name)

    return loss_reduced, predictions, metrics_score


