from typing import Union, Optional, Dict

import torch
import torch.nn as nn
from seg.registry.registry import MODELS
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmseg.models.utils.make_divisible import make_divisible

from monai.networks.layers.factories import Conv, Pool
from monai.networks.blocks import ADN
from typing import Sequence

@MODELS.register_module()
class EX_KD(nn.Module):
    def __init__(self,
                 in_channels: int,
                 ratio: int = 4):
        super().__init__()

        channels = in_channels // ratio

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=channels,
                kernel_size=(1, 1)),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(1, 3),
                padding=(0, 1)),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(3, 1),
                padding=(1, 0)),
            nn.Conv2d(
                in_channels=channels,
                out_channels=in_channels,
                kernel_size=(1, 1)))

        self.conv2 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        attention = self.conv1(x)
        attention = self.conv2(attention)
        return attention * x


class EX_KD_3D(nn.Module):
    def __init__(self,
                 spatial_dims: int,
                 in_channels: int,
                 # out_channels: int,
                 # strides: Union[Sequence[int], int] = 1,
                 kernel_size: Union[Sequence[int], int] = 3,
                 adn_ordering: str = "NDA",
                 act: Union[tuple, str, None] = "PRELU",
                 norm: Union[tuple, str, None] = "INSTANCE",
                 dropout: Union[tuple, str, float, None] = None,
                 dropout_dim: Optional[int] = 1,
                 bias: bool = True,
                 ratio: int = 4):
        super().__init__()

        if spatial_dims != 3:
            raise NotImplementedError
        if kernel_size != 3:
            raise NotImplementedError
        channels = in_channels // ratio

        self.receive_block = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=channels,
                kernel_size=(1, 1, 1)),
            nn.Conv3d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(1, 1, 3),
                padding=(0, 0, 1)),
            nn.Conv3d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(1, 3, 1),
                padding=(0, 1, 0)),
            nn.Conv3d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0)),
            nn.Conv3d(
                in_channels=channels,
                out_channels=in_channels,
                kernel_size=(1, 1, 1)))

        self.adn1 = ADN(
                ordering=adn_ordering,
                in_channels=in_channels,
                act=act,
                norm=norm,
                norm_dim=spatial_dims,
                dropout=dropout,
                dropout_dim=dropout_dim)

        self.convert_block = nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 1, 1))

        self.adn2 = ADN(
                ordering=adn_ordering,
                in_channels=in_channels,
                act='sigmoid',
                norm=norm,
                norm_dim=spatial_dims,
                dropout=dropout,
                dropout_dim=dropout_dim)

        # self.conv_x = nn.Conv3d(in_channels, out_channels, kernel_size, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        attention = self.receive_block(x)
        attention = self.adn1(attention)

        converted_attention = self.convert_block(attention)
        converted_attention = self.adn2(converted_attention)

        # res_x = self.conv_x(x)
        return converted_attention * x


class PixelWiseAttention(BaseModule):
    def __init__(
        self,
        student_channels: int,
        spatial_dims: int = 2,
        ratio: Optional[int] = None,
        init_cfg: Optional[Dict] = dict(type='Kaiming', layer='_ConvNd')   # noqa
    ) -> None:
        super().__init__(init_cfg)
        self.spatial_dims = spatial_dims
        self.conv_cfg = dict(type='Conv2d' if spatial_dims == 2 else 'Conv3d')

        self.ch_attn = nn.Sequential(
            Pool[Pool.ADAPTIVEAVG, spatial_dims](1),
            ConvModule(
                in_channels=student_channels,
                out_channels=make_divisible(student_channels // ratio, 8),
                kernel_size=1,
                stride=1,
                conv_cfg=self.conv_cfg,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=make_divisible(student_channels // ratio, 8),
                out_channels=student_channels,
                kernel_size=1,
                stride=1,
                conv_cfg=self.conv_cfg,
                act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0)))

        self.sp_attn = ConvModule(
                in_channels=1,
                out_channels=1,
                kernel_size=7,
                stride=1,
                padding=3,
                conv_cfg=self.conv_cfg,
                act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0))

    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        B, C, H, W, D = feature.shape
        channel_attention = self.ch_attn(feature)
        spatial_attention = self.sp_attn(torch.mean(feature, 1).unsqueeze(1))
        pixel_wise_attention = torch.bmm(
            channel_attention.view(B, C, -1), spatial_attention.view(B, 1, -1)).view(B, C, H, W, D)

        return pixel_wise_attention


class R2AModule(BaseModule):
    def __init__(
        self,
        student_channels: int,
        spatial_dims: int = 2,
        ratio: Optional[int] = 16,
        init_cfg: Optional[Dict] = dict(type='Kaiming', layer='_ConvNd')   # noqa
    ) -> None:
        super().__init__(init_cfg)
        self.spatial_dims = spatial_dims
        self.conv_cfg = dict(type='Conv2d' if spatial_dims == 2 else 'Conv3d')

        self.attn = PixelWiseAttention(
            student_channels,
            spatial_dims,
            ratio)

    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        attention = self.attn(feature)
        outputs = feature * attention
        return outputs
