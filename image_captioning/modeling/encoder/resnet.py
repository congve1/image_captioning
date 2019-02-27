"""
Variant of the resnet module that takes cfg as an argument.
Example usage. Strings may be specified in the config file.
    model = ResNet(
        "StemWithFixedBatchNorm",
        "BottleneckWithFixedBatchNorm",
        "ResNet50StagesTo4",
    )
Custom implementations may be written in user code and hooked in via the
`register_*` functions.
"""
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv2d, BatchNorm2d

from image_captioning.utils.registry import Registry
from image_captioning.layers.batch_norm import FrozenBatchNorm2d
from image_captioning.modeling.make_layers import group_norm

# ResNet sate specification
StageSpec = namedtuple(
    "StageSpec",
    [
        "index", # Index of the stage, eg 1, 2, ..., 5
        "block_count", #Number of residual blocks in the stage
        "return_features", # True => return the last feature map from this stage
    ]
)

# --------------------------------------------------------------------------------
# Standard ResNet models
# --------------------------------------------------------------------------------
# Resnet-50 (including all stages)
ResNet50StagesTo5 = tuple(
    StageSpec(index = i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, False), (4, 3, True))
)
# Resnet-50 up to stage 4(excludes stage 5)
ResNet50StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, True))
)

# Resnet-101 (including all stages)
ResNet101StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, False), (4, 3, True))
)

# Resnet-101 up to stage 4(excludes stage 5)
ResNet101StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False),(2, 4, False),(3, 23, True))
)

class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()
        # If we want to use the cfg in forward(), then we should make a copy
        # of it and store it for later use:
        #self.cfg = cfg.clone()
        # Translate string names to implementations
        stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC]
        stage_specs = _STAGE_SPECS[cfg.MODEL.ENCODER.CONV_BODY]
        transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRANS_FUNC]

        # construct the stem module
        self.stem = stem_module(cfg)

        # construct the specified ResNetStages
        num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
        stage2_bottleneck_channels = num_groups * width_per_group
        stage2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        self.stages = []
        self.return_features = {}
        for stage_spec in stage_specs:
            name = "layer" + str(stage_spec.index)
            stage2_relative_factcor = 2 ** (stage_spec.index - 1)
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factcor
            out_channels = stage2_out_channels * stage2_relative_factcor
            module = _make_stage(
                transformation_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage_spec.block_count,
                num_groups,
                cfg.MODEL.RESNETS.STRIDE_IN_1X1,
                first_stride=int(stage_spec.index > 1) + 1,
            )
            in_channels = out_channels
            self.add_module(name, module)
            self.stages.append(name)
            self.return_features[name] = stage_spec.return_features
        # Optionally freeze (requires_grad=False) parts of the encoder
        self._freeze_encoder(cfg.MODEL.ENCODER.FREEZE_CONV_BODY_AT)

        # self.use_fc = cfg.MODEL.ENCODER.USE_FC_FEATUES
        self.att_size = cfg.MODEL.ENCODER.ATT_SIZE
    
    def _freeze_encoder(self, freeze_at):
        if freeze_at < 0:
            return
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem
            else:
                m = getattr(self, 'layer'+str(stage_index))
            for p in m.parameters():
                p.requires_grad = False
    
    def forward(self, x):
        outputs = []
        x = self.stem(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
            if self.return_features[stage_name]:
                outputs.append(x)
        # size: batch_size x fetures_dim x att_size x att_size 
        att = F.adaptive_avg_pool2d(outputs[-1], [self.att_size, self.att_size])
        # if self.use_fc:
        fc = outputs[-1].mean(3).mean(2)  # batch_size x features_dim
        return fc, att


def _make_stage(
    transformation_module,
    in_channels,
    bottleneck_channels,
    out_channels,
    block_count,
    num_groups,
    stride_in_1x1,
    first_stride,
    dilation=1
):
    blocks = []
    stride = first_stride
    for _ in range(block_count):
        blocks.append(
            transformation_module(
                in_channels,
                bottleneck_channels,
                out_channels,
                num_groups,
                stride_in_1x1,
                stride,
                dilation=dilation
            )
        )
        stride = 1
        in_channels = out_channels
    return nn.Sequential(*blocks)


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_roups,
        stride_in_1x1,
        stride,
        dilation,
        norm_func,
    ):
        super(Bottleneck, self).__init__()
        self.downsample = None
        if in_channels != out_channels:
            down_stride = stride if dilation == 1 else 1
            self.downsample = nn.Sequential(
                Conv2d(in_channels, out_channels, kernel_size=1, 
                       stride=stride, bias=False),
                norm_func(out_channels),
            )
            for modules in [self.downsample,]:
                for l in modules.modules():
                    if isinstance(l, Conv2d):
                        nn.init.kaiming_uniform_(l.weight, a=1)
        if dilation > 1:
            stride = 1 # reset to be 1
        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
        )
        self.bn1 = norm_func(bottleneck_channels)
        # TODO: specify init for the above

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1,
            bias=False,
            groups=num_roups,
            dilation=dilation
        )
        self.bn2 = norm_func(bottleneck_channels)

        self.conv3 = Conv2d(
            bottleneck_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = norm_func(out_channels)
        for l in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu_(out)

        out0 = self.conv3(out)
        out = self.bn3(out0)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu_(out)

        return out


class BaseStem(nn.Module):
    def __init__(self, cfg, norm_func):
        super(BaseStem, self).__init__()

        out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS

        self.conv1 = Conv2d(
            3, out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_func(out_channels)

        for l in [self.conv1, ]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


class BottleneckWithBatchNorm(Bottleneck):
    def __init__(
            self,
            in_channels,
            bottleneck_channels,
            out_channels,
            num_groups=1,
            stride_in_1x1=True,
            stride=1,
            dilation=1,
    ):
        super(BottleneckWithBatchNorm, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_roups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=BatchNorm2d
        )


class StemWithBatchNorm(BaseStem):
    def __init__(self, cfg):
        super(StemWithBatchNorm, self).__init__(cfg, norm_func=BatchNorm2d)


class BottleneckWithFixedBatchNorm(Bottleneck):
    def __init__(
            self,
            in_channels,
            bottleneck_channels,
            out_channels,
            num_groups=1,
            stride_in_1x1=True,
            stride=1,
            dilation=1,
    ):
        super(BottleneckWithFixedBatchNorm, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_roups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=FrozenBatchNorm2d
        )


class StemWithFixedBatchNorm(BaseStem):
    def __init__(self, cfg):
        super(StemWithFixedBatchNorm, self).__init__(cfg, norm_func=FrozenBatchNorm2d)


class BottleneckWithGN(Bottleneck):
    def __init__(
            self,
            in_channels,
            bottleneck_channels,
            out_channels,
            num_groups=1,
            stride_in_1x1=True,
            stride=1,
            dilation=1,
    ):
        super(BottleneckWithGN, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_roups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=group_norm
        )


class StemWithGN(BaseStem):
    def __init__(self, cfg):
        super(StemWithGN, self).__init__(cfg, norm_func=BatchNorm2d)


_TRANSFORMATION_MODULES = Registry({
    "BottleneckWithBatchNorm": BottleneckWithBatchNorm,
    "BottleneckWithFixedBatchNorm": BottleneckWithFixedBatchNorm,
    "BottleneckWithGN": BottleneckWithGN,
})

_STEM_MODULES = Registry({
    "StemWithBatchNorm": StemWithBatchNorm,
    "StemWithFixedBatchNorm": StemWithFixedBatchNorm,
    "StemWithGN": StemWithGN
})

_STAGE_SPECS = Registry({
    "R-50-C4": ResNet50StagesTo4,
    "R-50-C5": ResNet50StagesTo5,
    "R-101-C5": ResNet101StagesTo5,
    'R-101-C4': ResNet101StagesTo4
})