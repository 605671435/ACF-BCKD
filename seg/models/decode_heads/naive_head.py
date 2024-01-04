# Copyright (c) OpenMMLab. All rights reserved.
from .decode_head import BaseDecodeHead


class NaiveHead(BaseDecodeHead):

    def forward(self, inputs):
        """Forward function."""
        output = inputs
        return output
