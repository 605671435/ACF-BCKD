# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule, Sequential

from seg.registry import MODELS
from mmpretrain.models.backbones import ConvNeXt as _ConvNeXt
from ..utils import DSA
class ConvNeXtBlock(BaseModule):
    """ConvNeXt Block.

    Args:
        in_channels (int): The number of input channels.
        dw_conv_cfg (dict): Config of depthwise convolution.
            Defaults to ``dict(kernel_size=7, padding=3)``.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        mlp_ratio (float): The expansion ratio in both pointwise convolution.
            Defaults to 4.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. More details can be found in the note.
            Defaults to True.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.

    Note:
        There are two equivalent implementations:

        1. DwConv -> LayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv;
           all outputs are in (N, C, H, W).
        2. DwConv -> LayerNorm -> Permute to (N, H, W, C) -> Linear -> GELU
           -> Linear; Permute back

        As default, we use the second to align with the official repository.
        And it may be slightly faster.
    """

    def __init__(self,
                 in_channels,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.dsa = DSA(in_channels=in_channels)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)),
            requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):

        def _inner_forward(x):
            shortcut = x
            x = self.dsa(x)

            if self.gamma is not None:
                x = x.mul(self.gamma.view(1, -1, 1, 1))

            x = shortcut + self.drop_path(x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x

@MODELS.register_module()
class DSA_ConvNeXt(_ConvNeXt):
    def __init__(self,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 with_cp=False,
                 layer_scale_init_value=1e-6,
                 **kwargs):
        super().__init__(drop_path_rate=drop_path_rate,
                         norm_cfg=norm_cfg,
                         with_cp=with_cp,
                         layer_scale_init_value=layer_scale_init_value,
                         **kwargs)

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    layer_scale_init_value=layer_scale_init_value,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

        self._freeze_stages()
