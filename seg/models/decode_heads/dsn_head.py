# Copyright (c) OpenMMLab. All rights reserved.
import torch
from ..utils import DSA

from seg.registry import MODELS
from .fcn_head import FCNHead
from .decode_head import BaseDecodeHead
from mmcv.cnn import ConvModule
from mmseg.models.utils import resize
@MODELS.register_module()
class DSNHead(FCNHead):
    def __init__(self,
                 ratio=2,
                 **kwargs):
        super().__init__(num_convs=2, **kwargs)
        self.ratio = ratio
        self.dsn_block = DSA(
            in_channels=self.channels,
            ratio=self.ratio)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs[0](x)
        output = self.dsn_block(output)
        output = self.convs[1](output)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output

@MODELS.register_module()
class MSDSNHead(BaseDecodeHead):

    def __init__(self, ratio=2, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.ratio = ratio
        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.dsn_block = DSA(
            in_channels=self.channels,
            ratio=self.ratio)

        self.align = ConvModule(
            self.channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        inputs = [
            resize(
                level,
                size=inputs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for level in inputs
        ]

        inputs = torch.cat(inputs, dim=1)
        # apply a conv block to squeeze feature map
        x = self.squeeze(inputs)
        # apply dsm
        x = self.dsn_block(x)

        # apply a conv block to align feature map
        output = self.align(x)
        output = self.cls_seg(output)
        return output
