# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.models.decode_heads.ham_head import Hamburger


class Ham(nn.Module):
    def __init__(self,
                 in_channels,
                 # ham_channels=512,
                 ham_kwargs=dict(),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        super().__init__()
        ham_channels = in_channels // 4
        self.squeeze = ConvModule(
            in_channels,
            ham_channels,
            3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.hamburger = Hamburger(ham_channels, ham_kwargs, **kwargs)

        self.align = ConvModule(
            ham_channels,
            in_channels,
            3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        """Forward function."""
        # apply a conv block to squeeze feature map
        x = self.squeeze(x)
        # apply hamburger module
        x = self.hamburger(x)
        # apply a conv block to align feature map
        output = self.align(x)
        return output

class Ham2(nn.Module):
    def __init__(self,
                 in_channels,
                 # ham_channels=512,
                 ham_kwargs=dict(),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        super().__init__()
        ham_channels = in_channels // 4
        self.squeeze = ConvModule(
            in_channels,
            ham_channels,
            1,
            # padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.hamburger = Hamburger(ham_channels, ham_kwargs, **kwargs)

        self.align = ConvModule(
            ham_channels,
            in_channels,
            1,
            # padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        """Forward function."""
        # apply a conv block to squeeze feature map
        x = self.squeeze(x)
        # apply hamburger module
        x = self.hamburger(x)
        # apply a conv block to align feature map
        output = self.align(x)
        return output

class Ham3(nn.Module):
    def __init__(self,
                 in_channels,
                 # ham_channels=512,
                 ham_kwargs=dict(),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        super().__init__()
        # ham_channels = in_channels // 4

        self.hamburger = Hamburger(in_channels, ham_kwargs, **kwargs)


    def forward(self, x):
        """Forward function."""

        output = self.hamburger(x)

        return output

class HamEx(nn.Module):
    def __init__(self,
                 in_channels,
                 # ham_channels=512,
                 ham_kwargs=dict(),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        super().__init__()
        ham_channels = in_channels // 4
        self.squeeze = ConvModule(
            in_channels,
            ham_channels,
            3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.hamburger = nn.Sequential(
            ConvModule(
                ham_channels, ham_channels, 1, norm_cfg=None, act_cfg=None),
            ConvModule(
                ham_channels, ham_channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        )

        self.align = ConvModule(
            ham_channels,
            in_channels,
            3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        """Forward function."""
        # apply a conv block to squeeze feature map
        x = self.squeeze(x)
        # apply hamburger module
        x = self.hamburger(x)
        # apply a conv block to align feature map
        output = self.align(x)
        return output
