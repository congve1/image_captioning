import torch
from torch import nn
from torch.nn import functional as F

from image_captioning.config import cfg


def get_group_gn(dim, dim_per_gp, num_groups):
    """
    get number of groups used by GroupNorm, based on number of channels
    Args:
        dim:
        dim_per_gp:
        num_groups:

    Returns:

    """

    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0, \
            "dim:{}, dim_per_gp: {}".format(dim, dim_per_gp)
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0, \
            "dim: {}, num_groups: {}".format(dim, num_groups)
        group_gn = num_groups

    return group_gn


def group_norm(out_channels, affine=True, divisor=1):
    out_channels = out_channels // divisor
    dim_per_gp = cfg.MODEL.GROUP_NORM.DIM_PER_GP // divisor
    num_groups = cfg.MODEL.GROUP_NORM.NUM_GROUPS // divisor

    eps = cfg.MODEL.GROUP_NORM.EPS
    return nn.GroupNorm(
        get_group_gn(out_channels, dim_per_gp, num_groups),
        out_channels,
        eps,
        affine
    )


def make_conv3x3(
        in_channels,
        out_channels,
        dilation=1,
        stride=1,
        use_gn=False,
        use_relu=False,
        kaiming_init=True,
):
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False if use_gn else True,
    )
    if kaiming_init:
        nn.init.kaiming_normal_(
            conv.weight, mode="fan_out", nonlinearity="relu"
        )
    else:
        nn.init.normal_(conv.weight, std=0.01)
    if not use_gn:
        nn.init.constant_(conv.bias, 0)
    module = [conv, ]
    if use_gn:
        module.append(group_norm(out_channels))
    if use_relu:
        module.append(nn.ReLU(inplace=True))
    if len(module) > 1:
        return nn.Sequential(*module)
    return conv


def make_fc(dim_in, hidden_dim, use_gn=False):
    if use_gn:
        fc = nn.Linear(dim_in, hidden_dim, bias=False)
        nn.init.kaiming_uniform_(fc.weight, a=1)
        return nn.Sequential(fc, group_norm(hidden_dim))
    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=1)
    nn.init.constant_(fc.bias, 0)
    return fc


def conv_with_kaiming_unifrom(use_gn=False, use_relu=False):
    def make_conv(
            in_channels, out_channels, kernel_size, stride=1, dilation=1
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size - 1),
            dilation=dilation,
            bias=False if use_gn else True
        )
        nn.init.kaiming_uniform_(conv.weight, a=1)
        if not use_gn:
            nn.init.constant_(conv.bias, 0)
        module = [conv, ]
        if use_gn:
            module.append(group_norm(out_channels))
        if use_relu:
            module.append(nn.ReLU(inplace=True))
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv
    return make_conv