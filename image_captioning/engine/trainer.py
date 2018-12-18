import datetime
import logging
import time

import torch.nn as nn
from image_captioning.utils.metric_logger import MetricLogger
from image_captioning.config import cfg
from image_captioning.modeling.utils import clip_gradients
from image_captioning.modeling.utils import LanguageModelCriterion


def do_train(
        model,
        train_data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        log_period,
        val_function,
        arguments
):
    logger = logging.getLogger('image_captioning.trainer')
    logger.info("Start training")
    meters = MetricLogger(delimiter='  ', name=cfg.SOLVER.METRIC_LOGGER_NAME)
    max_iter = len(train_data_loader)
    start_iter = arguments['iteration']
    model.train()
    start_training_time = time.time()
    ce_criterion = LanguageModelCriterion()
    criterion = ce_criterion
    best_cider_score = -1000.0
    end = time.time()
    for iteration, data in enumerate(train_data_loader, start_iter):
        data_time = time.time() - end
        end = time.time()
        iteration = iteration + 1
        arguments['iteration'] = iteration

        scheduler.step()

        att_features = data['att_features'].to(device)
        fc_features = data['fc_features'].to(device)
        cap_lens = data['cap_lens'].to(device)
        captions = data['captions'].to(device)

        outputs, weights = model(fc_features, att_features, captions)
        loss = criterion(outputs, captions[:, 1:], cap_lens+1)

        optimizer.zero_grad()
        loss.backward()
        #clip_gradients(optimizer, cfg.SOLVER.GRAD_CLIP)
        optimizer.step()

        batch_time = time.time() - end
        meters.update(loss=loss)
        meters.update(batch_time=batch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        if iteration % log_period == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]['lr'],
                )
            )
        #meters.add_scalar('loss', loss.item(), iteration)
        # save model and do evaluation
        if iteration % checkpoint_period == 0:
            checkpointer.save('model_{:07d}'.format(iteration), **arguments)
            val_loss, predictions, scores = val_function(model, device)
            logger.info("validation loss:{:.4f}".format(val_loss))
            meters.add_scalar('val_loss', val_loss, iteration)
            for metric, score in scores.items():
                logger.info("metric {}: {:.4f}".format(metric, score))
                meters.add_scalar(metric, score, iteration)
            model.train()
            if scores['CIDEr'] > best_cider_score:
                best_cider_score = scores['CIDEr']
                logger.info("best cider score: {:.4f}".format(best_cider_score))
                checkpointer.save('model_best')
        if iteration == max_iter:
            checkpointer.save('model_final', **arguments)
        end = time.time()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )


