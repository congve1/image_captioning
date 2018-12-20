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

def build_masks(batch_size, seq_len, cap_lens, device):
    masks = torch.zeros((batch_size, seq_len)).to(device)
    for i in range(batch_size):
        masks[i, :cap_lens[i]] = torch.ones(cap_lens[i])
    return masks

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, inputs, targets, cap_lens):
        """
        calculate word level cross entropy loss
        Args:
            inputs (torch(.cuda).FloatTensor): the input of the softmax, with size
            batch_size X seq_length X vocab_size
            targets (torch(.cuda).LongTensor): the ground truth word indexs, with
            size batch_size X seq_length
            cap_lens (torch(.cuda).LongTensor): the length of the captions.(with
            <end> word includes)

        Returns:
            output: cross entropy loss

        """
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        masks = build_masks(batch_size, seq_len, cap_lens, inputs.device)
        inputs = to_contiguous(inputs).view(-1, inputs.shape[-1])
        inputs = F.log_softmax(inputs, dim=1)
        targets = to_contiguous(targets).view(-1, 1)
        masks = to_contiguous(masks).view(-1, 1)
        output = - inputs.gather(1, targets) * masks
        output = torch.sum(output) / torch.sum(masks)
        return output


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, log_probs, rewards, cap_lens):
        batch_size = log_probs.size(0)
        seq_length = log_probs.size(1)
        masks = build_masks(
            batch_size, seq_length, cap_lens, log_probs.device
        ).view(-1)
        log_probs = log_probs.view(-1)
        rewards = rewards.view(-1)
        reward_loss = -log_probs * rewards * masks
        reward_loss = torch.sum(reward_loss) / torch.sum(masks)

        return reward_loss
