from einops import rearrange
from copy import deepcopy
from typing import List, Optional, Union
from torch import nn
import torch
import numpy as np

import torch.nn.functional

import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_3tuple, trunc_normal_

from mmengine.model import BaseModule
from seg.registry import MODELS

from ..utils.unet_2022 import *

@MODELS.register_module()
class UNet2022(BaseModule):
    def __init__(self,
                 pretrain_img_size=[224, 224],
                 patch_size=[4, 4],
                 in_chans=1,
                 embed_dim=96,
                 depths=[3, 3, 3, 3],
                 num_heads=[3, 6, 12, 24],
                 window_size=[7, 7, 14, 7],
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 init_cfg: Optional[dict] = None,
                 ):
        super().__init__(init_cfg=init_cfg)

        self.pretrain_img_size = pretrain_img_size

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices

        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    pretrain_img_size[0] // patch_size[0] // 2 ** i_layer,
                    pretrain_img_size[1] // patch_size[1] // 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_path=dpr[sum(
                    depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging
                if (i_layer < self.num_layers - 1) else None,
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def forward(self, x):
        """Forward function."""

        x = self.patch_embed(x)
        down = []

        Wh, Ww = x.size(2), x.size(3)

        x = self.pos_drop(x)

        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = x_out.permute(0, 2, 3, 1)
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()

                down.append(out)
        return down
