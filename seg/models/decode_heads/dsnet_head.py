# Copyright (c) OpenMMLab. All rights reserved.
import torch
from ..utils.dsa import DSA_V13, DSA_V14

from seg.registry import MODELS
from .fcn_head import FCNHead
from .decode_head import BaseDecodeHead
from mmcv.cnn import ConvModule
from mmseg.models.utils import resize
@MODELS.register_module()
class DSNetHead(BaseDecodeHead):
    def __init__(self,
                 dsnet_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 **kwargs):
        super().__init__(**kwargs)
        self.dsnet_block = DSA_V14(
            in_channels=self.channels,
            norm_cfg=dsnet_cfg)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.dsnet_block(x)
        output = self.cls_seg(output)
        return output

class DSNHeadV2(FCNHead):
    def __init__(self,
                 ratio=16,
                 **kwargs):
        super().__init__(num_convs=2, **kwargs)
        self.ratio = ratio
        self.dsn_block = DSA_V13(
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


