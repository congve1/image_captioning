from collections import defaultdict
from collections import deque

import torch
import tensorboardX


class SmoothedValue(object):
    """
    Track a series of values and provide access to smoothed values over a
    window or the global series average
    """
    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class MetricLogger(object):
    def __init__(self, delimiter='\t', log_period=20, name='image_captioning'):
        self.meters = defaultdict(lambda: SmoothedValue(log_period))
        self.delimiter = delimiter
        self.writer = tensorboardX.SummaryWriter(log_dir=('runs/'+name))

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' objects has no attribute '{}'".format(
                    type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.median, meter.global_avg)
            )
        return self.delimiter.join(loss_str)

    def add_scalar(self, tag, value, n_iter):
        self.writer.add_scalar(tag, value, n_iter)

    def add_histogram(self, tag, data, n_iter):
        self.writer.add_histogram(tag, data, n_iter)

    def close_writer(self):
        self.writer.close()
