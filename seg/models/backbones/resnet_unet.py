# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn
from dynamic_network_architectures.architectures.unet \
    import ResidualEncoderUNet as _ResidualEncoderUNet
from dynamic_network_architectures.architectures.unet \
    import PlainConvUNet as _PlainConvUNet

class ResidualEncoderUNet(_ResidualEncoderUNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for i in range(len(self.decoder.seg_layers)):
            self.decoder.seg_layers[i] = nn.Identity()

class PlainConvUNet(_PlainConvUNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for i in range(len(self.decoder.seg_layers)):
            self.decoder.seg_layers[i] = nn.Identity()