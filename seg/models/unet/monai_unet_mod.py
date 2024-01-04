# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import warnings
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm, Conv
# from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, export, SkipMode, look_up_option

__all__ = ["UNetMod"]


class SkipConnection(nn.Module):
    """
    Combine the forward pass input with the result from the given submodule::

        --+--submodule--o--
          |_____________|

    The available modes are ``"cat"``, ``"add"``, ``"mul"``.
    """

    def __init__(self, dim: int = 1, mode: str | SkipMode = "cat") -> None:
        """

        Args:
            submodule: the module defines the trainable branch.
            dim: the dimension over which the tensors are concatenated.
                Used when mode is ``"cat"``.
            mode: ``"cat"``, ``"add"``, ``"mul"``. defaults to ``"cat"``.
        """
        super().__init__()
        # self.submodule = submodule
        self.dim = dim
        self.mode = look_up_option(mode, SkipMode).value

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # y = self.submodule(x)

        if self.mode == "cat":
            return torch.cat([x, y], dim=self.dim)
        if self.mode == "add":
            return torch.add(x, y)
        if self.mode == "mul":
            return torch.mul(x, y)
        raise NotImplementedError(f"Unsupported mode {self.mode}.")


class UNetMod(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Sequence[int] | int = 3,
        up_kernel_size: Sequence[int] | int = 3,
        num_res_units: int = 0,
        act: tuple | str = Act.PRELU,
        norm: tuple | str = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        do_ds: bool = False
    ) -> None:
        super().__init__()

        if len(channels) < 3:
            raise ValueError("the length of `channels` should be no less than 3.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if isinstance(kernel_size, Sequence) and len(kernel_size) != spatial_dims:
            raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence) and len(up_kernel_size) != spatial_dims:
            raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        self.do_ds = do_ds

        # build model
        conv_type = Conv[Conv.CONV, spatial_dims]

        in_channels = [in_channels, *channels]
        out_channels = [*channels, out_channels]
        '''
        We start at second floor (capital 'DOWN'), then climb up. This is why we range from 'len(channels) - 2'.
        e.g. in_channels=[1(0), 32(1), 64(2), 128, 256], out_channels=[32(0), 64(1), 128(2), 256, 14(-1)], index order:[2, 1, 0]
        index=2, DOWN2: in=64, out=128, stride=2, Bottom: in=128, out=256, stride=1, Up2: in=128+256, out=64, stride=-2
        index=1, Down1: in=32, out=64,  stride=2, Up1: in=64+64, out=32, stride=-2
        index=0: Down0: in=1,  out=32,  stride=2, Up0: in=32+32, out=14, stride=-2
        i (in, out)                                                     (in, out)
        0 (  1, 32)   input->Down0+----------------------------o-Up0->  (  32+32, 14)
        1 ( 32, 64)               +Down1+-----------------o-Up1+        (  64+64, 32)
        2 ( 64, 128)                    +DOWN2+------o-Up2+             (128+256, 64)
        3 (128, 256)                          +bottom+
        '''
        for i in range(len(channels) - 2, -1, -1):
            if i == len(channels) - 2:
                upc = out_channels[i] + out_channels[i + 1]
                is_top = True
                self.bottom_layer = self._get_bottom_layer(out_channels[i], out_channels[i + 1])
                if self.do_ds:
                    seg_layer = conv_type(upc, out_channels[-1], kernel_size=1)
                    self.add_module(f'seg_layer{len(channels) - i - 1}', seg_layer)
            else:
                upc = out_channels[i] * 2
                is_top = False
                if self.do_ds:
                    seg_layer = conv_type(out_channels[i], out_channels[-1], kernel_size=1)
                    self.add_module(f'seg_layer{len(channels) - i - 1}', seg_layer)
            down_layer = self._get_down_layer(
                in_channels[i], out_channels[i], strides[i], is_top=i == 0)
            up_layer = self._get_up_layer(
                upc, out_channels[i - 1], strides[i], is_top=i == 0)
            self.add_module(f'down_layer{i + 1}', down_layer)
            self.add_module(f'up_layer{len(channels) - i - 1}', up_layer)
        # build connect module
        self.skip_connect = SkipConnection(dim=1, mode='cat')

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the encoding (down) part of a layer of the network. This typically will downsample data at some point
        in its structure. Its output is used as input to the next layer down and is concatenated with output from the
        next layer to form the input for the decode (up) part of the layer.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        mod: nn.Module
        if self.num_res_units > 0:
            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Returns the bottom or bottleneck layer at the bottom of the network linking encode to decode halves.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the decoding (up) part of a layer of the network. This typically will upsample data at some point
        in its structure. Its output is used as input to the next layer up.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        conv: Convolution | nn.Sequential

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        skips = []
        seg_outputs = []
        for i in range(len(self.strides)):
            down_layer = getattr(self, f'down_layer{i + 1}')
            x = down_layer(x)
            skips.append(x)
        x = self.bottom_layer(x)
        for i in range(len(self.strides)):
            x = self.skip_connect(x, skips[-1 - i])
            if i == 0 and self.do_ds:
                seg_layer = getattr(self, 'seg_layer1')
                seg_output = seg_layer(x)
                seg_outputs.append(seg_output)
            up_layer = getattr(self, f'up_layer{i + 1}')
            x = up_layer(x)
            if self.do_ds and i < len(self.strides) - 1:
                seg_layer = getattr(self, f'seg_layer{i + 2}')
                seg_output = seg_layer(x)
                seg_outputs.append(seg_output)
        if self.do_ds:
            # seg_outputs = [F.interpolate(seg_output, size=x.shape[2:],
            #                              mode='bilinear' if self.dimensions == 2 else 'trilinear')
            #                for seg_output in seg_outputs]
            seg_outputs = reversed(seg_outputs)
            return [x, *seg_outputs]
        else:
            return x


