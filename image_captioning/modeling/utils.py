import torch
from torch import nn
from torch.nn import functional as F
from torch._six import inf


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


def clip_gradients(optimizer, max_norm, norm_type=2):
    parameters = []
    for group in optimizer.param_groups:
        for param in group['params']:
            parameters.append(param)
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0.
        for p in parameters:
            param_norm = p.grad.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm


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
        inputs = F.log_softmax(inputs, dim=2)
        inputs = to_contiguous(inputs).view(-1, inputs.shape[-1])
        targets = to_contiguous(targets).view(-1, 1)
        masks = to_contiguous(masks).view(-1, 1)
        output = - inputs.gather(1, targets) * masks
        output = torch.sum(output) / torch.sum(masks)
        return output


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, log_probs, rewards, sample_seqs, vocab):
        batch_size = log_probs.size(0)
        seq_length = log_probs.size(1)
        end_idx = vocab['<end>']
        pad_idx = vocab['<pad>']
        masks = sample_seqs.new_ones(
            sample_seqs.size(), dtype=torch.float
        ).to(sample_seqs.device)
        #masks[sample_seqs==end_idx] = 0.
        masks[sample_seqs==pad_idx] = 0.
        masks = cat([masks.new(masks.size(0), 1).fill_(1), masks[:, :-1]], 1).reshape(-1)
        log_probs = log_probs.view(-1)
        rewards = rewards.view(-1)
        reward_loss = -log_probs * rewards * masks
        reward_loss = torch.sum(reward_loss) / torch.sum(masks)

        return reward_loss
