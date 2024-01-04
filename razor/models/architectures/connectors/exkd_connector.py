# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.utils import resize
from mmrazor.registry import MODELS
from mmrazor.models.architectures.connectors.base_connector import BaseConnector

from mmcv.cnn import ConvModule
from mmengine.utils import is_tuple_of

from mmseg.models.utils.make_divisible import make_divisible

from monai.networks.layers.factories import Conv, Pool


class SELayer(nn.Module):

    def __init__(self,
                 spatial_dims,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'),
                          dict(type='HSigmoid', bias=3.0, divisor=6.0))):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert is_tuple_of(act_cfg, dict)
        self.global_avgpool = Pool[Pool.ADAPTIVEAVG, spatial_dims](1)
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=make_divisible(channels // ratio, 8),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=make_divisible(channels // ratio, 8),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        # return x * out
        return out


@MODELS.register_module()
class EXKDConnector(BaseConnector):

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        spatial_dim: int = 2,
        student_shape: Optional[int] = None,
        teacher_shape: Optional[int] = None,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(init_cfg)
        if student_channels != teacher_channels:
            if spatial_dim == 2:
                conv = nn.Conv2d
            elif spatial_dim == 3:
                conv = nn.Conv3d
            else:
                raise TypeError
            self.align = conv(
                student_channels,
                teacher_channels,
                kernel_size=1,
                stride=1,
                padding=0)
        else:
            self.align = None
        self.student_shape = student_shape
        self.teacher_shape = teacher_shape

    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        if self.student_shape is not None and self.teacher_shape is not None:
            feature = resize(feature,
                             size=self.teacher_shape,
                             mode='nearest')
        if self.align is not None:
            feature = self.align(feature)

        return feature

@MODELS.register_module()
class AddConnector(BaseConnector):

    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        if self.align is not None:
            feature = self.align(feature)

        return feature


class R2AConnector(BaseConnector):

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        spatial_dims: int = 2,
        student_shape: Optional[int] = None,
        teacher_shape: Optional[int] = None,
        init_cfg: Optional[Dict] = dict(type='Kaiming', layer='_ConvNd')   # noqa
    ) -> None:
        super().__init__(init_cfg)
        self.spatial_dims = spatial_dims
        self.student_shape = student_shape
        self.teacher_shape = teacher_shape
        self.conv_cfg = dict(type='Conv2d' if spatial_dims == 2 else 'Conv3d')
        if self.student_shape is None and self.teacher_shape is None:
            self.do_interpolate = False
        else:
            self.do_interpolate = True if student_shape != teacher_shape else False
        # self.se_layer = SELayer(
        #     spatial_dims=spatial_dims,
        #     channels=student_channels,
        #     conv_cfg=self.conv_cfg)

        self.ch_attn = nn.Sequential(
            Pool[Pool.ADAPTIVEAVG, spatial_dims](1),
            ConvModule(
                in_channels=student_channels,
                out_channels=make_divisible(student_channels // 16, 8),
                kernel_size=1,
                stride=1,
                conv_cfg=self.conv_cfg,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=make_divisible(student_channels // 16, 8),
                out_channels=student_channels,
                kernel_size=1,
                stride=1,
                conv_cfg=self.conv_cfg,
                act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0)))

        self.sp_attn = nn.Sequential(
            ConvModule(
                in_channels=1,
                out_channels=1,
                kernel_size=7,
                stride=1,
                padding=3,
                conv_cfg=self.conv_cfg,
                act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0)))

        self.convertor = Conv[Conv.CONV, spatial_dims](
            in_channels=student_channels,
            out_channels=teacher_channels,
            kernel_size=1)

    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        B, C, H, W, D = feature.shape
        channel_attention = self.ch_attn(feature)
        spatial_attention = self.sp_attn(torch.mean(feature, 1).unsqueeze(1))
        pixel_wise_attention = torch.bmm(
            channel_attention.view(B, C, -1), spatial_attention.view(B, 1, -1)).view(B, C, H, W, D)
        residual = self.convertor(pixel_wise_attention)
        if self.do_interpolate:
            residual = F.interpolate(
                residual,
                size=self.teacher_shape,
                mode='bilinear' if self.spatial_dims == 2 else 'trilinear')
        return residual


class R2AConvertor(BaseConnector):

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        spatial_dims: int = 2,
        student_shape: Optional[int] = None,
        teacher_shape: Optional[int] = None,
        init_cfg: Optional[Dict] = dict(type='Kaiming', layer='_ConvNd')   # noqa
    ) -> None:
        super().__init__(init_cfg)
        self.spatial_dims = spatial_dims
        self.student_shape = student_shape
        self.teacher_shape = teacher_shape
        if self.student_shape is None and self.teacher_shape is None:
            self.do_interpolate = False
        else:
            self.do_interpolate = True if student_shape != teacher_shape else False

        self.convertor = Conv[Conv.CONV, spatial_dims](
            in_channels=student_channels,
            out_channels=teacher_channels,
            kernel_size=1)

    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        residual = self.convertor(feature)
        if self.do_interpolate:
            residual = F.interpolate(
                residual,
                size=self.teacher_shape,
                mode='bilinear' if self.spatial_dims == 2 else 'trilinear')
        return residual
