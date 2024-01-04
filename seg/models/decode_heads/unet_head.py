# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.utils import resize
from seg.registry import MODELS
from .decode_head import BaseDecodeHead

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
    ):
        super().__init__()
        self.conv1 = ConvModule(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x, up_size, skip=None):
        if x.shape[-2:] != up_size:
            x = resize(
                input=x,
                size=up_size,
                mode='bilinear',
                align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

@MODELS.register_module()
class UNetHead(BaseDecodeHead):

    def __init__(self,
                 in_index=(0, 1, 2, 3),
                 in_channels=(256, 512, 1024, 2048),
                 mid_channels=512,
                 skip=(True, True, True, True),
                 kernel_size=3,
                 concat_input=True,
                 **kwargs):
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        self.skip = skip
        super().__init__(in_index=in_index,
                         in_channels=in_channels,
                         **kwargs)
        self.convs = ConvModule(
            self.in_channels[-1],
            mid_channels,
            kernel_size=kernel_size,
            stride=2,
            # padding=1,
            # dilation=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        in_channels = (mid_channels, ) + in_channels[::-1]
        out_channels = in_channels[1:]

        blocks = []
        for i in in_index:
            in_ch = in_channels[i]
            out_ch = out_channels[i]
            sk_ch = in_channels[i + 1]
            blocks.append(DecoderBlock(in_ch, out_ch, sk_ch))

        self.blocks = nn.ModuleList(blocks)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        x.append(self.convs(x[-1]))
        stage_feats = x[::-1]
        for i, block in enumerate(self.blocks):
            x = stage_feats[i]
            skip = stage_feats[i + 1] if self.skip[i] else None
            x = block(x,
                      up_size=stage_feats[i + 1].shape[-2:],
                      skip=skip)
        return x

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
