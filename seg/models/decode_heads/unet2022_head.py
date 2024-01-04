# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from mmseg.utils import SampleList
from .decode_head import BaseDecodeHead
from mmseg.models.losses import accuracy
from ..utils.unet_2022 import BasicLayer_up, Patch_Expanding

class decoder(nn.Module):
    def __init__(self,
                 pretrain_img_size,
                 embed_dim,
                 patch_size=[4, 4],
                 depths=[3, 3, 3],
                 num_heads=[24, 12, 6],
                 window_size=[14, 7, 7],
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm
                 ):
        super().__init__()

        self.num_layers = len(depths)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers)[::-1]:
            layer = BasicLayer_up(
                dim=int(embed_dim * 2 ** (len(depths) - i_layer - 1)),
                input_resolution=(
                    pretrain_img_size[0] // patch_size[0] // 2 ** (len(depths) - i_layer - 1),
                    pretrain_img_size[1] // patch_size[1] // 2 ** (len(depths) - i_layer - 1)),

                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_path=dpr[sum(
                    depths[:(len(depths) - i_layer - 1)]):sum(depths[:(len(depths) - i_layer)])],
                norm_layer=norm_layer,
                upsample=Patch_Expanding
            )
            self.layers.append(layer)
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

    def forward(self, x, skips):
        outs = []
        H, W = x.size(2), x.size(3)
        x = self.pos_drop(x)

        for i in range(self.num_layers)[::-1]:
            layer = self.layers[i]
            x, H, W, = layer(x, skips[i], H, W)
            outs.append(x)
        return outs

@MODELS.register_module()
class UNet2022_Head(BaseDecodeHead):

    def __init__(self,
                 hidden_size: int = 768,
                 n_skip: int = 3,
                 skip_channels: list = [1024, 512, 256, 16],
                 decoder_channels: tuple = (256, 128, 64, 16),
                 norm_cfg: dict = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.decoder = decoder(
                               pretrain_img_size=self.crop_size,
                               window_size = self.window_size[::-1][1:],
                               embed_dim=self.embed_dim,
                               patch_size=self.patch_size,
                               depths=self.depths[::-1][1:],
                               num_heads=self.num_heads[::-1][1:]
                              )

    def forward(self, inputs):
        x, skips = self._transform_inputs(inputs)
        outputs = self.decoder(x, skips)
        output = self.cls_seg(outputs)
        return output
