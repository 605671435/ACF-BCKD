from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from monai.apps.utils import download_url
from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.squeeze_and_excitation import SEBottleneck, SEResNetBottleneck, SEResNeXtBottleneck
from monai.networks.layers.factories import Act, Conv, Dropout, Norm, Pool
from monai.utils.module import look_up_option
from monai.networks.nets.senet import SENet as _SENet
from monai.networks.nets.senet import _load_state_dict


class SENet(_SENet):
    def __init__(self, **kwargs) -> None:
        super(SENet, self).__init__(**kwargs)

        del self.adaptive_avg_pool, self.last_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x


class SEResNet50(SENet):
    """SEResNet50 based on `Squeeze-and-Excitation Networks` with optional pretrained support when spatial_dims is 2."""

    def __init__(
        self,
        layers: Sequence[int] = (3, 4, 6, 3),
        groups: int = 1,
        reduction: int = 16,
        dropout_prob: float | None = None,
        inplanes: int = 64,
        downsample_kernel_size: int = 1,
        input_3x3: bool = False,
        pretrained: bool = False,
        progress: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            block=SEResNetBottleneck,
            layers=layers,
            groups=groups,
            reduction=reduction,
            dropout_prob=dropout_prob,
            inplanes=inplanes,
            downsample_kernel_size=downsample_kernel_size,
            input_3x3=input_3x3,
            **kwargs,
        )
        if pretrained:
            # it only worked when `spatial_dims` is 2
            _load_state_dict(self, "se_resnet50", progress)