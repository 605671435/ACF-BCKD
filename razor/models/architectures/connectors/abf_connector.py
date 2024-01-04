# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from razor.registry import MODELS
from mmrazor.models.architectures.connectors.base_connector import BaseConnector

@MODELS.register_module()
class ABFConnector(BaseConnector):

    def __init__(self,
                 student_channels,
                 mid_channel,
                 teacher_channels,
                 student_shapes,
                 teacher_shapes,
                 fuse):
        super(ABFConnector, self).__init__()
        self.student_shapes = student_shapes
        self.teacher_shapes = teacher_shapes
        self.conv1 = nn.Sequential(
            nn.Conv2d(student_channels, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, teacher_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(teacher_channels),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                    nn.Conv2d(mid_channel*2, 2, kernel_size=1),
                    nn.Sigmoid(),
                )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward_train(self, x, y=None):
        n,_,h,w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, self.student_shapes, mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = (x * z[:,0].view(n,1,h,w) + y * z[:,1].view(n,1,h,w))
        # output
        y = self.conv2(x)
        y = F.interpolate(y, self.teacher_shapes, mode="nearest")
        return y, x

    def forward(self, x, y=None) -> torch.Tensor:

        return self.forward_train(x, y)
