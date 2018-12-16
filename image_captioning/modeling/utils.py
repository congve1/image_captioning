import torch
from torch import nn
from torch.nn import functional as F


def cat(tensors, dim=0):
    """
    Efficent version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


def clip_gradients(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def build_masks(self, batch_size, seq_len, cap_lens):
        masks = torch.zeros((batch_size, seq_len))
        for i in range(batch_size):
            masks[i, :cap_lens[i]] = torch.ones(cap_lens[i])
        return masks

    def forward(self, inputs, targets, cap_lens):
        batch_size = inputs.size(0)
        seq_size = inputs.size(1)
        masks = self.build_masks(batch_size, seq_size, cap_lens)
        inputs = to_contiguous(inputs).view(-1, inputs.shape[-1])
        inputs = F.log_softmax(inputs, dim=1)
        targets = to_contiguous(targets).view(-1, 1)
        masks = to_contiguous(masks).view(-1, 1)
        output = - inputs.gather(1, targets) * masks
        output = torch.sum(output) / torch.sum(masks)
        return output
