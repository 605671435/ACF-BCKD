# Copyright (c) OpenMMLab. All rights reserved.
# Originally from https://github.com/visual-attention-network/segnext
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
import torch.nn as nn
from mmseg.models.backbones import MSCAN as _MSCAN
from mmseg.models.backbones.mscan import MSCASpatialAttention as _MSCASpatialAttention
from mmseg.models.backbones.mscan import MSCABlock as _MSCABlock
from seg.registry import MODELS
from ..utils import DSA

class MSCASpatialAttention(_MSCASpatialAttention):
    def __init__(self,
                 in_channels, **kwargs):
        super().__init__(in_channels=in_channels, **kwargs)
        self.spatial_gating_unit = DSA(in_channels)

class MSCABlock(_MSCABlock):
    def __init__(self,
                 channels,
                 attention_kernel_sizes,
                 attention_kernel_paddings,
                 act_cfg,
                 use_dsa,
                 **kwargs):
        super().__init__(channels=channels,
                         attention_kernel_sizes=attention_kernel_sizes,
                         attention_kernel_paddings=attention_kernel_paddings,
                         act_cfg=act_cfg,
                         **kwargs)
        self.use_dsa = use_dsa
        if use_dsa:
            self.attn = MSCASpatialAttention(in_channels=channels,
                                             attention_kernel_sizes=attention_kernel_sizes,
                                             attention_kernel_paddings=attention_kernel_paddings,
                                             act_cfg=act_cfg)
        else:
            self.attn = None

    def forward(self, x, H, W):
        """Forward function."""

        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        if self.use_dsa:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
                self.attn(self.norm1(x)))
        else:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
            self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x

@MODELS.register_module()
class DSN_MSCAN(_MSCAN):
    def __init__(self,
                 in_channels=3,
                 depths=[3, 4, 6, 3],
                 embed_dims=[64, 128, 256, 512],
                 attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
                 mlp_ratios=[4, 4, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 dsa_stages=(False, False, False, True),
                 **kwargs):
        super().__init__(in_channels=in_channels,
                         depths=depths,
                         embed_dims=embed_dims,
                         attention_kernel_sizes=attention_kernel_sizes,
                         attention_kernel_paddings=attention_kernel_paddings,
                         mlp_ratios=mlp_ratios,
                         drop_rate=drop_rate,
                         drop_path_rate=drop_path_rate,
                         act_cfg=act_cfg,
                         norm_cfg=norm_cfg,
                         **kwargs)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0
        for i in range(self.num_stages):
            block = nn.ModuleList([
                MSCABlock(
                    channels=embed_dims[i],
                    attention_kernel_sizes=attention_kernel_sizes,
                    attention_kernel_paddings=attention_kernel_paddings,
                    mlp_ratio=mlp_ratios[i],
                    drop=drop_rate,
                    drop_path=dpr[cur + j],
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    use_dsa=dsa_stages[i]) for j in range(depths[i])
            ])
            cur += depths[i]

            setattr(self, f'block{i + 1}', block)
