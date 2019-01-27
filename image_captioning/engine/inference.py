import datetime
import logging
import time
import os

import torch
from tqdm import tqdm

from image_captioning.utils.miscellaneous import decode_sequence
from image_captioning.data.datasets.evaluation import coco_eval


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
            outputs, weights = model(fc_features, att_fatures, captions)
            loss = criterion(outputs, captions[:, 1:], cap_lens+1)
            val_loss = loss.item()
            val_loss_count += 1
            val_loss_sum += val_loss
            seqs, seq_log_probs, weights = model.decode_search(
                fc_features, att_fatures, beam_size=beam_size
            )
            sents = decode_sequence(vocab, seqs)
            for k, sent in enumerate(sents):
                entry = {'image_id': cocoids[k], 'caption': sent}
                if cocoids[k] not in done_ids:
                    done_ids[cocoids[k]] = 0
                    predictions.append(entry)
    return predictions, val_loss_sum / val_loss_count


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
    logger = logging.getLogger("image_captioning.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images)".format(dataset_name, len(dataset)))
    start_time = time.time()
    predictions, loss = compute_on_dataset(
        model, criterion, data_loader, vocab, beam_size, device, logger,
    )

    metrics_score = coco_eval(predictions, dataset_name)

    return loss, predictions, metrics_score


