from torch import nn
from torch.nn import BatchNorm2d, ReLU
from mmcv.cnn import ConvModule


class AAM(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 norm_cfg=dict(type=BatchNorm2d, requires_grad=True),
                 act_cfg=dict(type=ReLU, inplace=True)):
        super(AAM, self).__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv2 = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, padding=0),
            nn.Softmax(dim=1))

        self.conv4 = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv_out = ConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, input_high, input_low):
        mid_high = self.global_pooling(input_high)
        weight_high = self.conv1(mid_high)

        mid_low = self.global_pooling(input_low)
        weight_low = self.conv2(mid_low)

        weight = self.conv3(weight_low + weight_high)
        low = self.conv4(input_low)
        out = input_high + low.mul(weight)
        out = self.conv_out(out)
        return out
