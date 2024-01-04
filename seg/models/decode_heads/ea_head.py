# Copyright (c) OpenMMLab. All rights reserved.
import torch
from ..utils import External_Attention

from seg.registry import MODELS
from .fcn_head import FCNHead

@MODELS.register_module()
class EAHead(FCNHead):
    def __init__(self,
                 ratio=2,
                 **kwargs):
        super().__init__(num_convs=2, **kwargs)
        self.ratio = ratio
        self.ea_block = External_Attention(
            dim=self.channels,
            in_dim=self.channels)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs[0](x)

        output = self.ea_block(output)
        output = self.convs[1](output)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output
