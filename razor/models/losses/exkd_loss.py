# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union, Optional, Dict
import torch
from torch import nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from razor.registry import MODELS
from mmrazor.models.losses import L2Loss, ATLoss
from mmseg.models.utils.make_divisible import make_divisible

from monai.networks.layers.factories import Conv, Pool

from seg.models.utils.se_layer import SELayer
from seg.models.utils.gcnet import ContextBlock
from seg.models.utils import CBAM
from seg.models.utils.hamburger import Ham

@MODELS.register_module()
class EXKD_Loss(L2Loss):

    def forward(
            self,
            s_feature: torch.Tensor,
            t_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Forward computation.

        Args:
            s_feature (torch.Tensor): The student model feature with
                shape (N, C, H, W) or shape (N, C).
            t_feature (torch.Tensor): The teacher model feature with
                shape (N, C, H, W) or shape (N, C).
        """
        t_feature = t_feature.contiguous()

        if self.normalize:
            s_feature = self.normalize_feature(s_feature)
            t_feature = self.normalize_feature(t_feature)

        loss = torch.sum(torch.pow(torch.sub(s_feature, t_feature), 2))

        # Calculate l2_loss as dist.
        if self.dist:
            loss = torch.sqrt(loss)
        else:
            if self.div_element:
                loss = loss / s_feature.numel()
            else:
                loss = loss / s_feature.size(0)

        return self.loss_weight * loss


# class PixelWiseAttention(BaseModule):
class PixelWiseAttention(nn.Module):
    def __init__(
        self,
        student_channels: int,
        spatial_dims: int = 2,
        ratio: Optional[int] = None,
        init_cfg: Optional[Dict] = dict(type='Kaiming', layer=['_ConvNd', 'Conv2d', 'Conv3d'])
    ) -> None:
        # super().__init__(init_cfg=init_cfg)
        super().__init__()
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

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        B, C, H, W, D = feature.shape
        channel_attention = self.ch_attn(feature)
        spatial_attention = self.sp_attn(torch.mean(feature, 1).unsqueeze(1))
        pixel_wise_attention = torch.bmm(
            channel_attention.view(B, C, -1), spatial_attention.view(B, 1, -1)).view(B, C, H, W, D)

        return pixel_wise_attention


class EXKDV2_Loss(BaseModule):
    def __init__(self,
                 student_channels: int,
                 teacher_channels: Optional[int] = None,
                 spatial_dims: int = 2,
                 ratio: Optional[int] = 16,
                 student_shape: Optional[int] = None,
                 teacher_shape: Optional[int] = None,
                 alpha: Optional[float] = 1.0,
                 beta: Optional[float] = 1.0,
                 loss_weight: Optional[float] = 1.0,
                 at_weight: Optional[float] = 25000,
                 attn_type: Optional[str] = 'pw',
                 init_cfg: Optional[Dict] = dict(type='Kaiming', layer=['_ConvNd', 'Conv2d', 'Conv3d'])):
        super().__init__(init_cfg=init_cfg)
        self.spatial_dims = spatial_dims
        self.conv_cfg = dict(type='Conv2d' if spatial_dims == 2 else 'Conv3d')
        self.student_shape = student_shape
        self.teacher_shape = teacher_shape
        self.alpha = alpha
        self.beta = beta
        self.loss_weight = loss_weight

        if attn_type == 'pw':
            self.attn = PixelWiseAttention(
                student_channels,
                spatial_dims,
                ratio)
        elif attn_type == 'se':
            self.attn = SELayer(
                in_channels=student_channels,
                ratio=ratio,
                conv_cfg=dict(type='Conv3d'))
        elif attn_type == 'gcnet':
            self.attn = ContextBlock(
                in_channels=student_channels,
                ratio=ratio)
        elif attn_type == 'cbam':
            self.attn = CBAM(
                in_channels=student_channels,
                reduction_ratio=ratio)
        else:
            raise TypeError(f'wrong attention type {attn_type}')

        if self.student_shape is None and self.teacher_shape is None:
            self.do_interpolate = False
        else:
            self.do_interpolate = True if self.student_shape != self.teacher_shape else False

        # self.convertor1 = Conv[Conv.CONV, spatial_dims](
        #     in_channels=student_channels,
        #     out_channels=teacher_channels,
        #     kernel_size=1)

        self.convertor2 = Conv[Conv.CONV, spatial_dims](
            in_channels=student_channels,
            out_channels=student_channels,
            kernel_size=1)

        # self.convertor1 = ConvModule(
        #     in_channels=student_channels,
        #     out_channels=teacher_channels,
        #     kernel_size=1,
        #     conv_cfg=self.conv_cfg,
        #     # norm_cfg=dict(type=f'IN{self.spatial_dims}d'),
        #     # act_cfg=dict(type='PReLU')
        # )
        #
        # self.convertor2 = ConvModule(
        #     in_channels=student_channels,
        #     out_channels=teacher_channels,
        #     kernel_size=1,
        #     conv_cfg=self.conv_cfg,
        #     # norm_cfg=dict(type=f'IN{self.spatial_dims}d'),
        #     # act_cfg=dict(type='PReLU')
        # )

        # self.loss = L2Loss()
        self.loss = ATLoss(loss_weight=at_weight)
        # self.loss = sim_dis_compute

    def forward(
            self,
            s_feature: torch.Tensor,
            t_feature: torch.Tensor,
            t_residual: torch.Tensor
    ) -> torch.Tensor:
        """Forward computation.

        """
        s_attention = self.attn(s_feature)
        s_feature = s_feature * s_attention
        # s_feature = self.convertor1(s_feature)
        s_residual = self.convertor2(s_attention)

        if self.do_interpolate is True:
            s_feature = F.interpolate(
                s_feature,
                size=self.teacher_shape,
                mode='nearest')
            s_residual = F.interpolate(
                s_residual,
                size=self.teacher_shape,
                mode='nearest')

        loss1 = self.loss(s_feature, t_feature)
        loss2 = self.loss(s_residual, t_residual)
        # loss1 = F.mse_loss(s_feature, t_feature, reduction="mean")
        # loss2 = F.mse_loss(s_residual, t_residual, reduction="mean")
        loss = self.alpha * loss1 + self.beta * loss2
        return self.loss_weight * loss


class Attention_Loss(BaseModule):
    def __init__(self,
                 student_channels: int,
                 teacher_channels: int,
                 spatial_dims: int = 2,
                 ratio: Optional[int] = 16,
                 student_shape: Optional[int] = None,
                 teacher_shape: Optional[int] = None,
                 alpha: Optional[float] = 1.0,
                 beta: Optional[float] = 1.0,
                 loss_weight: Optional[float] = 1.0,
                 at_weight: Optional[float] = 25000,
                 attn_type: Optional[str] = 'pw',
                 init_cfg: Optional[Dict] = dict(type='Kaiming', layer=['_ConvNd', 'Conv2d', 'Conv3d'])):
        super().__init__(init_cfg=init_cfg)
        self.spatial_dims = spatial_dims
        self.conv_cfg = dict(type='Conv2d' if spatial_dims == 2 else 'Conv3d')
        self.student_shape = student_shape
        self.teacher_shape = teacher_shape
        self.alpha = alpha
        self.beta = beta
        self.loss_weight = loss_weight

        if attn_type == 'pw':
            self.attn = PixelWiseAttention(
                student_channels,
                spatial_dims,
                ratio)
        elif attn_type == 'se':
            self.attn = SELayer(
                in_channels=student_channels,
                ratio=ratio,
                conv_cfg=dict(type='Conv3d'))
        elif attn_type == 'gcnet':
            self.attn = ContextBlock(
                in_channels=student_channels,
                ratio=ratio)
        elif attn_type == 'cbam':
            self.attn = CBAM(
                in_channels=student_channels,
                reduction_ratio=ratio)
        else:
            raise TypeError(f'wrong attention type {attn_type}')

        if self.student_shape is None and self.teacher_shape is None:
            self.do_interpolate = False
        else:
            self.do_interpolate = True if self.student_shape != self.teacher_shape else False

        # self.convertor = Conv[Conv.CONV, spatial_dims](
        #     in_channels=student_channels,
        #     out_channels=student_channels,
        #     kernel_size=1)

        self.loss = ATLoss(loss_weight=at_weight)

    def forward(
            self,
            s_feature: torch.Tensor,
            t_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Forward computation.

        """
        s_attention = self.attn(s_feature)
        s_feature = s_feature * s_attention
        # s_feature = self.convertor(s_feature)

        if self.do_interpolate is True:
            s_feature = F.interpolate(
                s_feature,
                size=self.teacher_shape,
                mode='nearest')

        loss = self.loss(s_feature, t_feature)
        return self.loss_weight * loss


class EXKDV3_Loss(BaseModule):
    def __init__(self,
                 student_channels: int,
                 teacher_channels: int,
                 spatial_dims: int = 2,
                 ratio: Optional[int] = 16,
                 student_shape: Optional[int] = None,
                 teacher_shape: Optional[int] = None,
                 alpha: Optional[float] = 1.0,
                 beta: Optional[float] = 1.0,
                 loss_weight: Optional[float] = 1.0,
                 init_cfg: Optional[Dict] = dict(type='Kaiming', layer=['_ConvNd', 'Conv2d', 'Conv3d'])):
        super().__init__(init_cfg=init_cfg)
        self.spatial_dims = spatial_dims
        self.conv_cfg = dict(type='Conv2d' if spatial_dims == 2 else 'Conv3d')
        self.student_shape = student_shape
        self.teacher_shape = teacher_shape
        self.alpha = alpha
        self.beta = beta
        self.loss_weight = loss_weight

        self.attn = PixelWiseAttention(
            student_channels,
            spatial_dims,
            ratio)

        if self.student_shape is None and self.teacher_shape is None:
            self.do_interpolate = False
        else:
            self.do_interpolate = True if self.student_shape != self.teacher_shape else False

        # self.convertor1 = Conv[Conv.CONV, spatial_dims](
        #     in_channels=student_channels,
        #     out_channels=teacher_channels,
        #     kernel_size=1)

        # self.convertor2 = Conv[Conv.CONV, spatial_dims](
        #     in_channels=student_channels,
        #     out_channels=student_channels,
        #     kernel_size=1)

        # self.convertor1 = ConvModule(
        #     in_channels=student_channels,
        #     out_channels=teacher_channels,
        #     kernel_size=1,
        #     conv_cfg=self.conv_cfg,
        #     # norm_cfg=dict(type=f'IN{self.spatial_dims}d'),
        #     # act_cfg=dict(type='PReLU')
        # )
        #
        self.convertor2 = ConvModule(
            in_channels=student_channels,
            out_channels=teacher_channels,
            kernel_size=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=dict(type=f'IN{self.spatial_dims}d'),
            act_cfg=dict(type='PReLU')
        )

        # self.loss = L2Loss()
        self.loss = ATLoss(loss_weight=25000)
        # self.loss = sim_dis_compute

    def forward(
            self,
            s_feature: torch.Tensor,
            t_feature: torch.Tensor,
            t_residual: torch.Tensor
    ) -> torch.Tensor:
        """Forward computation.

        """
        s_attention = self.attn(s_feature)
        s_feature = s_feature * s_attention
        # s_feature = self.convertor1(s_feature)
        s_residual = self.convertor2(s_attention)

        if self.do_interpolate is True:
            s_feature = F.interpolate(
                s_feature,
                size=self.teacher_shape,
                mode='nearest')
            s_residual = F.interpolate(
                s_residual,
                size=self.teacher_shape,
                mode='nearest')

        loss1 = self.loss(s_feature, t_feature)
        loss2 = self.loss(s_residual, t_residual)
        # loss1 = F.mse_loss(s_feature, t_feature, reduction="mean")
        # loss2 = F.mse_loss(s_residual, t_residual, reduction="mean")
        loss = self.alpha * loss1 + self.beta * loss2
        return self.loss_weight * loss