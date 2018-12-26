import math
from bisect import bisect_right

import torch


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=1.0/3,
            warmup_iters=500,
            warmup_method='linear',
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of"
                " increasing integers. Got {}".format(milestones)
            )
        if warmup_method not in ('constant', 'linear'):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)
        if self.last_epoch == -1:
            self.last_epoch += 1

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            elif self.warmup_method == 'linear':
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
                base_lr
                * warmup_factor
                * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs
        ]

class SGDRCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            T_max,
            T_multi=2,
            eta_min = 0.,
            warmup_factor=1.0/3,
            warmup_iters=500,
            warmup_method='linear',
            last_epoch=-1,
    ):
        if warmup_method not in ('constant', 'linear'):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.T_max = T_max
        self.eta_min = eta_min
        self.T_multi = T_multi
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(SGDRCosineLR, self).__init__(optimizer, last_epoch)
        if self.last_epoch == -1:
            self.last_epoch += 1

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            elif self.warmup_method == 'linear':
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [
                base_lr
                * warmup_factor
                for base_lr in self.base_lrs
         ]
        else:
            if (self.last_epoch - self.warmup_iters) > self.T_max:
                self.last_epoch = self.warmup_iters
                self.T_max = self.T_max * self.T_multi
            return [
                self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi
                              * (self.last_epoch-self.warmup_iters)
                                        /(self.T_max)))/2
                for base_lr in self.base_lrs
            ]

