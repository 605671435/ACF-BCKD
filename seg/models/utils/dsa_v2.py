import torch
import torch.nn as nn
from mmengine.model import kaiming_init
from mmcv.cnn import ConvModule


class SelectiveKernelAttn(nn.Module):
    def __init__(self, channels, num_paths=2, attn_channels=32, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        # https://github.com/rwightman/pytorch-image-models/blob/709d5e0d9d2d3f501531506eda96a435737223a3/timm/layers/selective_kernel.py
        """ Selective Kernel Attention Module
        Selective Kernel attention mechanism factored out into its own module.
        """
        super(SelectiveKernelAttn, self).__init__()
        self.num_paths = num_paths
        self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
        self.bn = norm_layer(attn_channels)
        self.act = act_layer(inplace=True)
        self.fc_select = nn.Conv2d(attn_channels, channels * num_paths, kernel_size=1, bias=False)

        # self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.fc_reduce, mode='fan_in')
        kaiming_init(self.fc_select, mode='fan_in')
        self.fc_reduce.inited = True
        self.fc_select.inited = True

    def forward(self, x):
        # [B, 2, C, H, W]
        assert x.shape[1] == self.num_paths

        # [B, C, 1, 1]
        x = x.sum(1).mean((2, 3), keepdim=True)

        # [B, IC, 1, 1]
        x = self.fc_reduce(x)
        # [B, IC, 1, 1]
        x = self.bn(x)
        # [B, IC, 1, 1]
        x = self.act(x)
        # [B, C * 2, 1, 1]
        x = self.fc_select(x)

        B, C, H, W = x.shape
        # [B, 2, C, 1, 1]
        x = x.view(B, self.num_paths, C // self.num_paths, H, W)
        # [B, 2, C, 1, 1]
        x = torch.softmax(x, dim=1)
        return x


class DSA(nn.Module):
    __name__ = 'dsa_block'

    def __init__(self,
                 in_channels: int,
                 attn_types: tuple = ('ch', 'sp'),
                 fusion_type: str = 'dsa',
                 ratio: int = 2,
                 norm_cfg: dict = None,
                 bias: bool = False):
        super().__init__()
        assert isinstance(attn_types, (list, tuple))
        valid_attn_types = ['ch', 'sp']
        assert all([a in valid_attn_types for a in attn_types])
        assert len(attn_types) > 0, 'at least one attention should be used'
        assert isinstance(fusion_type, str)
        valid_fusion_types = ['sq', 'pr', 'dsa', 'dam', 'None']
        assert fusion_type in valid_fusion_types
        if fusion_type == 'None':
            assert len(attn_types) == 1, f'Got fusion type is {fusion_type}, ' \
                                         f'{fusion_type} fusion only need one attention type, ' \
                                         f'but got attn_types is {attn_types}.'
        else:
            assert len(attn_types) == 2, f'Got fusion type is {fusion_type}' \
                                         f'and fusion need two attention types, but got attn_types are {attn_types}.'
        self.in_channels = in_channels
        self.channels = in_channels // ratio
        self.attn_types = attn_types
        self.fusion_type = fusion_type
        if 'ch' in attn_types:
            self.ch_attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvModule(
                    in_channels=self.in_channels,
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    # norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU')),
                ConvModule(
                    in_channels=self.channels,
                    out_channels=self.in_channels,
                    kernel_size=1,
                    stride=1,
                    # norm_cfg=norm_cfg,
                    act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0)))
        if 'sp' in attn_types:
            self.sp_attn = nn.Sequential(
                ConvModule(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=7,
                    stride=1,
                    padding=3,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0)))
        if fusion_type in ['dsa', 'dam']:
            self.sk = SelectiveKernelAttn(channels=self.in_channels, attn_channels=self.channels)
            if fusion_type == 'dsa':
                self.resConv = ConvModule(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=1,
                    # norm_cfg=dict(type='BN'),
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU'))
        if fusion_type in ['pr', 'sq', 'dam']:
            self.combine_attn = ConvModule(
                self.in_channels,
                self.in_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0))
        self.reset_parameters()

    def reset_parameters(self):
        if 'ch' in self.attn_types:
            kaiming_init(self.ch_attn, mode='fan_in')
            self.ch_attn.inited = True
        if 'sp' in self.attn_types:
            kaiming_init(self.sp_attn, mode='fan_in')
            self.sp_attn.inited = True
        if self.fusion_type == 'dsa':
            kaiming_init(self.resConv, mode='fan_in')
            self.resConv.inited = True
        if self.fusion_type in ['pr', 'sq', 'dam']:
            kaiming_init(self.combine_attn, mode='fan_in')
            self.combine_attn.inited = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        # [B, C, H, W]
        b, c, h, w = x.size()

        if 'ch' in self.attn_types:
            # [B, C, 1, 1]
            # channel_attn = torch.sigmoid(self.channel_attention(x))
            channel_attn = self.ch_attn(x)
            # print(f'channel_attn: {channel_attn.reshape(b, c)}')
            if 'sp' not in self.attn_types:
                out = x * channel_attn
                return out
            # [B, C, 1]
            channel_attn = channel_attn.reshape(b, c, 1)
        else:
            channel_attn = None
        if 'sp' in self.attn_types:
            # [B, 1, H * W]
            spatial_attn = self.sp_attn(torch.mean(x, 1).unsqueeze(1))
            # print(f'spatial_attn: {spatial_attn.reshape(b, h * w)}')
            if 'ch' not in self.attn_types:
                spatial_attn = spatial_attn.reshape(b, 1, h, w)
                out = x * spatial_attn
                return out
            spatial_attn = spatial_attn.reshape(b, 1, h * w)
        else:
            spatial_attn = None
        if self.fusion_type == 'sq' or self.fusion_type in ['dsa', 'dam']:
            # [B, C, H * W]
            sequence = torch.bmm(channel_attn, spatial_attn)
            # sequence = sequence.reshape(b, c * h * w)
            # sequence = torch.softmax(sequence, 1)
            # [B, C, H, W]
            sequence = sequence.reshape(b, c, h, w)
            if self.fusion_type not in ['dsa', 'dam']:
                out = x * self.combine_attn(sequence)
                return out
        else:
            sequence = None
        if self.fusion_type == 'pr' or self.fusion_type in ['dsa', 'dam']:
            # [B, C, H * W]
            parallel = spatial_attn + channel_attn
            # parallel = parallel.reshape(b, c * h * w)
            # parallel = torch.softmax(parallel, 1)
            # [B, C, H, W]
            parallel = parallel.reshape(b, c, h, w)
            if self.fusion_type not in ['dsa', 'dam']:
                out = x * self.combine_attn(parallel)
                return out
            else:
                # [B, 2, C, H, W]
                stack = torch.stack([sequence, parallel], dim=1)
                # [B, 2, C, 1, 1]
                sk_attn = self.sk(stack)
                # [B, 2, C, H, W]
                sq_pr = stack * sk_attn
                # [B, C, H, W]
                sq_pr = torch.sum(sq_pr, dim=1)
                if self.fusion_type == 'dam':
                    return x * self.combine_attn(sq_pr)
                # [B, C, H, W]
                # sq_pr = self.sk(sequence, parallel)
                # [B, C * H * W]
                sq_pr = sq_pr.reshape(b, c, h * w)

                self_attn = torch.softmax(sq_pr, 2)
                # [B, C, H, W]
                self_attn = self_attn.reshape(b, c, h, w)
                value = self.resConv(x)

                value = value * self_attn
                # [B, C, 1, 1]
                aggregation = value.reshape(b, c, h * w).sum(dim=2).reshape(b, c, 1, 1)
                # pool = self.avg_pool(self_attn).reshape(b, c, 1, 1)
                out = aggregation

                return out + x

class DSA_V7(nn.Module):
    __name__ = 'dsa_block'

    def __init__(self,
                 in_channels: int,
                 attn_types: tuple = ('ch', 'sp'),
                 fusion_type: str = 'dsa',
                 ratio: int = 2,
                 norm_cfg: dict = None,
                 bias: bool = False):
        super().__init__()
        assert isinstance(attn_types, (list, tuple))
        valid_attn_types = ['ch', 'sp']
        assert all([a in valid_attn_types for a in attn_types])
        assert len(attn_types) > 0, 'at least one attention should be used'
        assert isinstance(fusion_type, str)
        valid_fusion_types = ['sq', 'pr', 'dsa', 'dam', 'None']
        assert fusion_type in valid_fusion_types
        if fusion_type == 'None':
            assert len(attn_types) == 1, f'Got fusion type is {fusion_type}, ' \
                                         f'{fusion_type} fusion only need one attention type, ' \
                                         f'but got attn_types is {attn_types}.'
        else:
            assert len(attn_types) == 2, f'Got fusion type is {fusion_type}' \
                                         f'and fusion need two attention types, but got attn_types are {attn_types}.'
        self.in_channels = in_channels
        self.channels = in_channels // ratio
        self.attn_types = attn_types
        self.fusion_type = fusion_type
        if 'ch' in attn_types:
            self.ch_attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvModule(
                    in_channels=self.in_channels,
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    # norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU')),
                ConvModule(
                    in_channels=self.channels,
                    out_channels=self.in_channels,
                    kernel_size=1,
                    stride=1,
                    # norm_cfg=norm_cfg,
                    act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0)))
        if 'sp' in attn_types:
            self.sp_attn = nn.Sequential(
                ConvModule(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=7,
                    stride=1,
                    padding=3,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0)))
        if fusion_type in ['dsa', 'dam']:
            self.sk = SelectiveKernelAttn(channels=self.in_channels, attn_channels=self.channels)
            if fusion_type == 'dsa':
                self.resConv = ConvModule(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=1,
                    # norm_cfg=dict(type='BN'),
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU'))
                self.conv_out = nn.Sequential(
                    nn.Conv2d(self.in_channels, self.channels, kernel_size=1),
                    nn.LayerNorm([self.channels, 1, 1]),
                    nn.ReLU(inplace=True),  # yapf: disable
                    nn.Conv2d(self.channels, self.in_channels, kernel_size=1))
        if fusion_type in ['pr', 'sq', 'dam']:
            self.combine_attn = ConvModule(
                self.in_channels,
                self.in_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0))
        self.reset_parameters()

    def reset_parameters(self):
        if 'ch' in self.attn_types:
            kaiming_init(self.ch_attn, mode='fan_in')
            self.ch_attn.inited = True
        if 'sp' in self.attn_types:
            kaiming_init(self.sp_attn, mode='fan_in')
            self.sp_attn.inited = True
        if self.fusion_type == 'dsa':
            kaiming_init(self.resConv, mode='fan_in')
            kaiming_init(self.conv_out, mode='fan_in')
            self.resConv.inited = True
        if self.fusion_type in ['pr', 'sq', 'dam']:
            kaiming_init(self.combine_attn, mode='fan_in')
            self.combine_attn.inited = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        # [B, C, H, W]
        b, c, h, w = x.size()

        if 'ch' in self.attn_types:
            # [B, C, 1, 1]
            # channel_attn = torch.sigmoid(self.channel_attention(x))
            channel_attn = self.ch_attn(x)
            # print(f'channel_attn: {channel_attn.reshape(b, c)}')
            if 'sp' not in self.attn_types:
                out = x * channel_attn
                return out
            # [B, C, 1]
            channel_attn = channel_attn.reshape(b, c, 1)
        else:
            channel_attn = None
        if 'sp' in self.attn_types:
            # [B, 1, H * W]
            spatial_attn = self.sp_attn(torch.mean(x, 1).unsqueeze(1))
            # print(f'spatial_attn: {spatial_attn.reshape(b, h * w)}')
            if 'ch' not in self.attn_types:
                spatial_attn = spatial_attn.reshape(b, 1, h, w)
                out = x * spatial_attn
                return out
            spatial_attn = spatial_attn.reshape(b, 1, h * w)
        else:
            spatial_attn = None
        if self.fusion_type == 'sq' or self.fusion_type in ['dsa', 'dam']:
            # [B, C, H * W]
            sequence = torch.bmm(channel_attn, spatial_attn)
            # sequence = sequence.reshape(b, c * h * w)
            # sequence = torch.softmax(sequence, 1)
            # [B, C, H, W]
            sequence = sequence.reshape(b, c, h, w)
            if self.fusion_type not in ['dsa', 'dam']:
                out = x * self.combine_attn(sequence)
                return out
        else:
            sequence = None
        if self.fusion_type == 'pr' or self.fusion_type in ['dsa', 'dam']:
            # [B, C, H * W]
            parallel = spatial_attn + channel_attn
            # parallel = parallel.reshape(b, c * h * w)
            # parallel = torch.softmax(parallel, 1)
            # [B, C, H, W]
            parallel = parallel.reshape(b, c, h, w)
            if self.fusion_type not in ['dsa', 'dam']:
                out = x * self.combine_attn(parallel)
                return out
            else:
                # [B, 2, C, H, W]
                stack = torch.stack([sequence, parallel], dim=1)
                # [B, 2, C, 1, 1]
                sk_attn = self.sk(stack)
                # [B, 2, C, H, W]
                sq_pr = stack * sk_attn
                # [B, C, H, W]
                sq_pr = torch.sum(sq_pr, dim=1)
                if self.fusion_type == 'dam':
                    return x * self.combine_attn(sq_pr)
                # [B, C, H, W]
                # sq_pr = self.sk(sequence, parallel)
                # [B, C * H * W]
                sq_pr = sq_pr.reshape(b, c, h * w)

                self_attn = torch.softmax(sq_pr, 2)
                # [B, C, H, W]
                self_attn = self_attn.reshape(b, c, h, w)
                value = self.resConv(x)

                value = value * self_attn
                # [B, C, 1, 1]
                aggregation = value.reshape(b, c, h * w).sum(dim=2).reshape(b, c, 1, 1)
                # pool = self.avg_pool(self_attn).reshape(b, c, 1, 1)
                out = self.conv_out(aggregation)

                return out + x
