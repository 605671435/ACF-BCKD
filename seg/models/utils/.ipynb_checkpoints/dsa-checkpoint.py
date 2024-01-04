import torch
import torch.nn as nn
from mmengine.model import kaiming_init
from mmseg.models.utils.make_divisible import make_divisible
from seg.registry.registry import MODELS

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

class SK_Module(nn.Module):
    def __init__(self,
                 in_channels: int,
                 bias: bool = False):
        super(SK_Module, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        channels = in_channels // 2
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, in_channels, kernel_size=1, bias=bias))

    def forward(self, x1, x2):
        # [B, C, H, W]
        u = self.gap(x1 + x2)
        u = self.bottleneck(u)
        softmax_a = torch.softmax(u, 1)
        out = x1 * softmax_a + x2 * (1 - softmax_a)
        return out

@MODELS.register_module()
class DSA(nn.Module):
    def __init__(self,
                 in_channels: int,
                 attn_types: tuple = ('ch', 'sp'),
                 fusion_type: str = 'dsa',
                 ratio: int = 2,
                 bias: bool = False):
        super().__init__()
        assert isinstance(attn_types, (list, tuple))
        valid_attn_types = ['ch', 'sp']
        assert all([a in valid_attn_types for a in attn_types])
        assert len(attn_types) > 0, 'at least one attention should be used'
        assert isinstance(fusion_type, str)
        valid_fusion_types = ['sq', 'pr', 'dsa', 'None']
        assert fusion_type in valid_fusion_types
        if fusion_type == 'None':
            assert len(attn_types) == 1, 'None fusion only need one attention type.'
        else:
            assert len(attn_types) == 2, 'Fusion need two attention types.'
        self.in_channels = in_channels
        self.channels = in_channels // ratio
        self.attn_types = attn_types
        self.fusion_type = fusion_type
        if 'ch' in attn_types:
            self.conv_q_right = nn.Conv2d(self.in_channels, 1, kernel_size=1, bias=bias)
            self.conv_v_right = nn.Conv2d(self.in_channels, self.channels, kernel_size=1, bias=bias)
            # self.conv_up = nn.Conv2d(self.channels, self.in_channels,  kernel_size=1, bias=bias)
            self.conv_up = nn.Sequential(
                nn.LayerNorm(normalized_shape=[self.channels, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.channels, self.in_channels,  kernel_size=1, bias=bias),
                # nn.Softmax(dim=1)
            )
            self.softmax_right = nn.Softmax(dim=2)
        if 'sp' in attn_types:
            self.conv_q_left = nn.Conv2d(self.in_channels, self.channels, kernel_size=1, bias=bias)
            self.conv_v_left = nn.Conv2d(self.in_channels, self.channels, kernel_size=1, bias=bias)
            self.gap = nn.AdaptiveMaxPool2d(output_size=1)
        if fusion_type == 'dsa':
            self.sk = SelectiveKernelAttn(channels=self.in_channels, attn_channels=self.channels)
            # self.sk = SK_Module(in_channels=in_channels)
            self.resConv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels,  kernel_size=1, bias=bias),
                nn.ReLU(inplace=True)
            )
            # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.reset_parameters()

    def reset_parameters(self):
        if 'ch' in self.attn_types:
            kaiming_init(self.conv_q_right, mode='fan_in')
            kaiming_init(self.conv_v_right, mode='fan_in')
            self.conv_q_right.inited = True
            self.conv_v_right.inited = True
        if 'sp' in self.attn_types:
            kaiming_init(self.conv_q_left, mode='fan_in')
            kaiming_init(self.conv_v_left, mode='fan_in')
            self.conv_q_left.inited = True
            self.conv_v_left.inited = True
        if self.fusion_type == 'dsa':
            kaiming_init(self.resConv, mode='fan_in')
            self.resConv.inited = True

    def channel_attention(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial Pooling (psa)
        b, c, h, w = x.size()
        # [B, IC, H, W]
        input_x = self.conv_v_right(x)
        # [B, IC, H * W]
        input_x = input_x.reshape(b, self.channels, h * w)
        # [B, 1, H, W]
        context_mask = self.conv_q_right(x)
        # [B, 1, H * W]
        context_mask = context_mask.reshape(b, 1, h * w)
        context_mask = torch.softmax(context_mask, 2)
        # [B, H * W, 1]
        context_mask = context_mask.transpose(1, 2)
        # [B, IC, 1, 1]
        context = torch.matmul(input_x, context_mask).reshape(b, self.channels, 1, 1)
        # [B, C, 1, 1]
        channel_attn = self.conv_up(context)
        channel_attn = torch.sigmoid(channel_attn)
        return channel_attn

    def spatial_attention(self, x: torch.Tensor) -> torch.Tensor:
        # Channel Pooling (psa)
        b, c, h, w = x.size()
        # [B, IC, H, W]
        g_x = self.conv_q_left(x)
        # [B, IC, 1]
        avg_x = torch.softmax(self.gap(g_x), 1)
        # [B, 1, IC]
        avg_x = avg_x.reshape(b, 1, self.channels)
        # [B, IC, H * W]
        theta_x = self.conv_v_left(x)
        theta_x = theta_x.reshape(b, self.channels, h * w)
        # [B, 1, H * W]
        context = torch.matmul(avg_x, theta_x)
        # [B, 1, H * W]
        spatial_attn = torch.sigmoid(context)
        # [B, 1, H * W]
        # spatial_attn = torch.softmax(context, 2)

        return spatial_attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        # [B, C, H, W]
        b, c, h, w = x.size()

        if 'ch' in self.attn_types:
            # [B, C, 1, 1]
            # channel_attn = torch.sigmoid(self.channel_attention(x))
            channel_attn = self.channel_attention(x)
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
            spatial_attn = self.spatial_attention(x)
            # print(f'spatial_attn: {spatial_attn.reshape(b, h * w)}')
            if 'ch' not in self.attn_types:
                spatial_attn = spatial_attn.reshape(b, 1, h, w)
                out = x * spatial_attn
                return out
        else:
            spatial_attn = None
        if self.fusion_type == 'sq' or self.fusion_type == 'dsa':
            # [B, C, H * W]
            sequence = torch.bmm(channel_attn, spatial_attn)
            # sequence = sequence.reshape(b, c * h * w)
            # sequence = torch.softmax(sequence, 1)
            # [B, C, H, W]
            sequence = sequence.reshape(b, c, h, w)
            if self.fusion_type != 'dsa':
                out = x * sequence
                return out
        else:
            sequence = None
        if self.fusion_type == 'pr' or self.fusion_type == 'dsa':
            # [B, C, H * W]
            parallel = spatial_attn + channel_attn
            # parallel = parallel.reshape(b, c * h * w)
            # parallel = torch.softmax(parallel, 1)
            # [B, C, H, W]
            parallel = parallel.reshape(b, c, h, w)
            if self.fusion_type != 'dsa':
                out = x * parallel
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

                # [B, C, H, W]
                # sq_pr = self.sk(sequence, parallel)
                # [B, C * H * W]
                sq_pr = sq_pr.reshape(b, c, h * w)
                self_attn = torch.softmax(sq_pr, 2)
                # [B, C, H, W]
                self_attn = self_attn.reshape(b, c, h, w)
                value = self.resConv(x)

                # import seg.utils.misc
                # # channel_reduction = 'select_max'
                # channel_reduction = 'squeeze_mean'
                # seg.utils.misc.show_featmaps(x * channel_attn.reshape(b, c, 1, 1), channel_reduction=channel_reduction)
                # seg.utils.misc.show_featmaps(spatial_attn.reshape(b, 1, h, w), channel_reduction=channel_reduction)
                # seg.utils.misc.show_featmaps(sequence, channel_reduction=channel_reduction)
                # seg.utils.misc.show_featmaps(parallel, channel_reduction=channel_reduction)
                # seg.utils.misc.show_featmaps(self_attn[0], channel_reduction=channel_reduction)
                # seg.utils.misc.show_featmaps(value[0], channel_reduction=channel_reduction)

                # import seg.utils.misc
                # attention = seg.utils.misc.show_featmaps(self_attn[0], channel_reduction=None, topk=100,
                #                                          arrangement=(10, 10))
                # featmaps = seg.utils.misc.show_featmaps(value[0], channel_reduction=None, topk=100,
                #                                         arrangement=(10, 10))
                # import mmcv
                # featmaps == mmcv.bgr2rgb(featmaps)
                # mmcv.imwrite(featmaps, './output/featmaps.jpg')
                # mmcv.imwrite(attention, './output/attention.jpg')

                value = value * self_attn
                # [B, C, 1, 1]
                aggregation = value.reshape(b, c, h * w).sum(dim=2).reshape(b, c, 1, 1)
                # pool = self.avg_pool(self_attn).reshape(b, c, 1, 1)
                out = aggregation

                return out + x

@MODELS.register_module()
class DSA_ex(nn.Module):
    def __init__(self,
                 in_channels: int,
                 attn_types: tuple = ('ch', 'sp'),
                 fusion_type: str = 'dsa',
                 ratio: int = 2,
                 bias: bool = False):
        super().__init__()
        assert isinstance(attn_types, (list, tuple))
        valid_attn_types = ['ch', 'sp']
        assert all([a in valid_attn_types for a in attn_types])
        assert len(attn_types) > 0, 'at least one attention should be used'
        assert isinstance(fusion_type, str)
        valid_fusion_types = ['sq', 'pr', 'dsa', 'None']
        assert fusion_type in valid_fusion_types
        if fusion_type == 'None':
            assert len(attn_types) == 1, 'None fusion only need one attention type.'
        else:
            assert len(attn_types) == 2, 'Fusion need two attention types.'
        self.in_channels = in_channels
        self.channels = make_divisible(in_channels // ratio, 8)
        self.attn_types = attn_types
        self.fusion_type = fusion_type
        if 'ch' in attn_types:
            self.gap_ch = nn.AdaptiveAvgPool2d(1)
            self.conv1_ch = nn.Sequential(
                nn.Conv2d(self.in_channels, self.channels,  kernel_size=1, bias=bias),
                nn.ReLU(inplace=True),
            )
            self.conv2_ch = nn.Sequential(
                nn.Conv2d(self.channels, self.in_channels,  kernel_size=1, bias=bias),
                nn.Sigmoid(),
            )
        if 'sp' in attn_types:
            self.conv_sp = nn.Conv2d(self.in_channels, 1, kernel_size=1, bias=bias)
        if fusion_type == 'dsa':
            self.sk = SelectiveKernelAttn(channels=self.in_channels, attn_channels=self.channels)
            # self.sk = SK_Module(in_channels=in_channels)
            self.resConv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        if 'ch' in self.attn_types:
            kaiming_init(self.conv1_ch, mode='fan_in')
            kaiming_init(self.conv2_ch, mode='fan_in')
            self.conv1_ch.inited = True
            self.conv2_ch.inited = True
        if 'sp' in self.attn_types:
            kaiming_init(self.conv_sp, mode='fan_in')
            self.conv_sp.inited = True
        if self.fusion_type == 'dsa':
            kaiming_init(self.resConv, mode='fan_in')
            self.resConv.inited = True

    def channel_attention(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial Pooling (psa)
        out = self.gap_ch(x)
        out = self.conv1_ch(out)
        out = self.conv2_ch(out)
        return out

    def spatial_attention(self, x: torch.Tensor) -> torch.Tensor:
        # Channel Pooling (psa)
        b, c, h, w = x.size()
        # [B, C, H, W]
        out = self.conv_sp(x)
        # [B, 1, H, W]
        out = out.reshape(b, 1, h * w)
        out = torch.softmax(out, 2)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        # [B, C, H, W]
        b, c, h, w = x.size()

        if 'ch' in self.attn_types:
            # [B, C, 1, 1]
            # channel_attn = torch.sigmoid(self.channel_attention(x))
            channel_attn = self.channel_attention(x)
            # print(f'channel_attn: {channel_attn.reshape(b, c)}')
            if 'sp' not in self.attn_types:
                out = x * channel_attn
                return out + x
            # [B, C, 1]
            channel_attn = channel_attn.reshape(b, c, 1)
        else:
            channel_attn = None
        if 'sp' in self.attn_types:
            # [B, 1, H * W]
            spatial_attn = self.spatial_attention(x)
            if 'ch' not in self.attn_types:
                out = x * spatial_attn
                return out + x
        else:
            spatial_attn = None
        if self.fusion_type == 'sq' or self.fusion_type == 'dsa':
            # [B, C, H * W]
            sequence = torch.bmm(channel_attn, spatial_attn)
            # sequence = sequence.reshape(b, c * h * w)
            # sequence = torch.softmax(sequence, 1)
            # [B, C, H, W]
            sequence = sequence.reshape(b, c, h, w)
            if self.fusion_type != 'dsa':
                out = x * sequence
                return out + x
        else:
            sequence = None
        if self.fusion_type == 'pr' or self.fusion_type == 'dsa':
            # [B, C, H * W]
            parallel = spatial_attn + channel_attn
            # parallel = parallel.reshape(b, c * h * w)
            # parallel = torch.softmax(parallel, 1)
            # [B, C, H, W]
            parallel = parallel.reshape(b, c, h, w)
            if self.fusion_type != 'dsa':
                out = x * parallel
                return out + x
            else:

                # [B, 2, C, H, W]
                stack = torch.stack([sequence, parallel], dim=1)
                # [B, 2, C, 1, 1]
                sk_attn = self.sk(stack)
                # [B, 2, C, H, W]
                sq_pr = stack * sk_attn
                # [B, C, H, W]
                sq_pr = torch.sum(sq_pr, dim=1)

                # [B, C, H, W]
                # sq_pr = self.sk(sequence, parallel)
                # [B, C * H * W]
                sq_pr = sq_pr.reshape(b, c * h * w)
                self_attn = torch.softmax(sq_pr, 1)
                # [B, C, H, W]
                self_attn = self_attn.reshape(b, c, h, w)
                value = self.resConv(x)

                # import seg.utils.misc
                # seg.utils.misc.show_featmaps(self_attn[0])
                # seg.utils.misc.show_featmaps(value[0])

                self_attn = self_attn * value
                # [B, C, 1, 1]
                self_attn = self_attn.reshape(b, c, h * w).sum(dim=2).reshape(b, c, 1, 1)
                out = self_attn

                return out + x

@MODELS.register_module()
class EX_Module(nn.Module):
    def __init__(self,
                 in_channels: int,
                 attn_types: tuple = ('ch', 'sp'),
                 fusion_type: str = 'dsa',
                 ratio: int = 2,
                 bias: bool = False):
        super().__init__()
        assert isinstance(attn_types, (list, tuple))
        valid_attn_types = ['ch', 'sp']
        assert all([a in valid_attn_types for a in attn_types])
        assert len(attn_types) > 0, 'at least one attention should be used'
        assert isinstance(fusion_type, str)
        valid_fusion_types = ['sq', 'pr', 'dsa', 'None']
        assert fusion_type in valid_fusion_types
        if fusion_type == 'None':
            assert len(attn_types) == 1, 'None fusion only need one attention type.'
        else:
            assert len(attn_types) == 2, 'Fusion need two attention types.'
        self.in_channels = in_channels
        self.channels = in_channels // ratio
        self.attn_types = attn_types
        self.fusion_type = fusion_type
        if 'ch' in attn_types:
            self.conv_q_right = nn.Conv2d(self.in_channels, 1, kernel_size=1, bias=bias)
            self.conv_v_right = nn.Conv2d(self.in_channels, self.channels, kernel_size=1, bias=bias)
            self.conv_up = nn.Sequential(
                nn.LayerNorm(normalized_shape=[self.channels, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.channels, self.in_channels,  kernel_size=1, bias=bias),
            )
            self.softmax_right = nn.Softmax(dim=2)
        if 'sp' in attn_types:
            self.conv_q_left = nn.Conv2d(self.in_channels, self.channels, kernel_size=1, bias=bias)
            self.conv_v_left = nn.Conv2d(self.in_channels, self.channels, kernel_size=1, bias=bias)
            self.gap = nn.AdaptiveMaxPool2d(output_size=1)
        if fusion_type == 'dsa':
            self.sk = SK_Module(in_channels=in_channels)
            self.resConv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        if 'ch' in self.attn_types:
            kaiming_init(self.conv_q_right, mode='fan_in')
            kaiming_init(self.conv_v_right, mode='fan_in')
            self.conv_q_right.inited = True
            self.conv_v_right.inited = True
        if 'sp' in self.attn_types:
            kaiming_init(self.conv_q_left, mode='fan_in')
            kaiming_init(self.conv_v_left, mode='fan_in')
            self.conv_q_left.inited = True
            self.conv_v_left.inited = True
        if self.fusion_type == 'dsa':
            kaiming_init(self.resConv, mode='fan_in')
            self.resConv.inited = True

    def channel_attention(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial Pooling (psa)
        b, c, h, w = x.size()
        # [B, IC, H, W]
        input_x = self.conv_v_right(x)
        # [B, IC, H * W]
        input_x = input_x.reshape(b, self.channels, h * w)
        # [B, 1, H, W]
        context_mask = self.conv_q_right(x)
        # [B, 1, H * W]
        context_mask = context_mask.reshape(b, 1, h * w)
        context_mask = torch.softmax(context_mask, 2)
        # [B, H * W, 1]
        context_mask = context_mask.transpose(1, 2)
        # [B, IC, 1, 1]
        context = torch.matmul(input_x, context_mask).reshape(b, self.channels, 1, 1)
        # [B, C, 1, 1]
        channel_attn = self.conv_up(context)

        return channel_attn

    def spatial_attention(self, x: torch.Tensor) -> torch.Tensor:
        # Channel Pooling (psa)
        b, c, h, w = x.size()
        # [B, IC, H, W]
        g_x = self.conv_q_left(x)
        # [B, IC, 1]
        avg_x = torch.softmax(self.gap(g_x), 1)
        # [B, 1, IC]
        avg_x = avg_x.reshape(b, 1, self.channels)
        # [B, IC, H * W]
        theta_x = self.conv_v_left(x)
        theta_x = theta_x.reshape(b, self.channels, h * w)
        # [B, 1, H * W]
        context = torch.matmul(avg_x, theta_x)
        # [B, 1, H * W]
        # spatial_attn = torch.sigmoid(context)
        # [B, 1, H * W]
        spatial_attn = torch.softmax(context, 2)

        return spatial_attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        # [B, C, H, W]
        b, c, h, w = x.size()

        if 'ch' in self.attn_types:
            # [B, C, 1, 1]
            channel_attn = torch.sigmoid(self.channel_attention(x))
            if 'sp' not in self.attn_types:
                out = x * torch.softmax(channel_attn, 1)
                return out + x
            # [B, C, 1]
            channel_attn = channel_attn.reshape(b, c, 1)
        else:
            channel_attn = None
        if 'sp' in self.attn_types:
            # [B, 1, H * W]
            spatial_attn = self.spatial_attention(x)
            if 'ch' not in self.attn_types:
                spatial_attn = torch.softmax(spatial_attn, 2)
                spatial_attn = spatial_attn.reshape(b, 1, h, w)
                out = x * spatial_attn
                return out + x
        else:
            spatial_attn = None
        if self.fusion_type == 'sq' or self.fusion_type == 'dsa':
            # [B, C, H * W]
            sequence = torch.bmm(channel_attn, spatial_attn)
            # sequence = spatial_attn * channel_attn
            # [B, C, H, W]
            sequence = sequence.reshape(b, c, h, w)
            if self.fusion_type != 'dsa':
                out = x * sequence
                return out + x
        else:
            sequence = None
        if self.fusion_type == 'pr' or self.fusion_type == 'dsa':
            # [B, C, H * W]
            parallel = spatial_attn + channel_attn
            # [B, C, H, W]
            parallel = parallel.reshape(b, c, h, w)
            if self.fusion_type != 'dsa':
                out = x * parallel
                return out + x
            else:
                # [B, C, H, W]
                sq_pr = self.sk(sequence, parallel)
                # [B, C, H * W]
                sq_pr = sq_pr.reshape(b, c, h * w)
                self_attn = torch.softmax(sq_pr, 2)
                # [B, C, H, W]
                self_attn = self_attn.reshape(b, c, h, w)
                value = self.resConv(x)
                self_attn = self_attn * value
                # [B, C, 1, 1]
                self_attn = self_attn.reshape(b, c, h * w).sum(dim=2).reshape(b, c, 1, 1)
                out = self_attn

                return out + x