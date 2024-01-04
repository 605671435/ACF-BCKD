# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.utils import is_tuple_of

from mmseg.models.utils.make_divisible import make_divisible
from dynamic_network_architectures.building_blocks.regularization \
    import make_divisible as _make_divisible


class SELayer(nn.Module):
    """Squeeze-and-Excitation Module.

    Args:
        in_channels (int): The input (and output) channels of the SE layer.
        ratio (int): Squeeze ratio in SELayer, the intermediate channel will be
            ``int(channels/ratio)``. Default: 16.
        conv_cfg (None or dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configured
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configured by the first dict and the
            second activation layer will be configured by the second dict.
            Default: (dict(type='ReLU'), dict(type='HSigmoid', bias=3.0,
            divisor=6.0)).
    """

    def __init__(self,
                 in_channels,
                 ratio=16,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'),
                          dict(type='HSigmoid', bias=3.0, divisor=6.0))):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert is_tuple_of(act_cfg, dict)
        self.global_avgpool = nn.AdaptiveAvgPool3d(1)
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=make_divisible(in_channels // ratio, 8),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=make_divisible(in_channels // ratio, 8),
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out

class SqueezeExcite(nn.Module):
    """
    This class is taken from the timm package (https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/squeeze_excite.py)
    and slightly modified so that the convolution type can be adapted.

    SE Module as defined in original SE-Nets with a few additions
    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * global max pooling can be added to the squeeze aggregation
        * customizable activation, normalization, and gate layer
    """
    def __init__(
            self, in_channels, conv_op=nn.Conv2d, rd_ratio=1. / 16, rd_channels=None, rd_divisor=8, add_maxpool=False,
            act_layer=nn.ReLU, norm_layer=None, gate_layer=nn.Sigmoid):
        super(SqueezeExcite, self).__init__()
        self.add_maxpool = add_maxpool
        if not rd_channels:
            rd_channels = _make_divisible(in_channels * rd_ratio, rd_divisor, round_limit=0.)
        self.fc1 = conv_op(in_channels, rd_channels, kernel_size=1, bias=True)
        self.bn = norm_layer(rd_channels) if norm_layer else nn.Identity()
        self.act = act_layer(inplace=True)
        self.fc2 = conv_op(rd_channels, in_channels, kernel_size=1, bias=True)
        self.gate = gate_layer()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)