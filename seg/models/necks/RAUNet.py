import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
from torch.nn import BatchNorm2d
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer

class AAM(nn.Module):
    def __init__(self, in_ch, out_ch, norm_cfg, act_cfg):
        super(AAM, self).__init__()
        # norm_cfg = dict(type=BatchNorm2d, requires_grad=True)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, 1, padding=0),
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU(inplace=True))

        self.conv1 = ConvModule(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, 1, padding=0),
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU(inplace=True))

        self.conv2 = ConvModule(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, padding=0),
            nn.Softmax(dim=1))

        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, 1, padding=0),
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU(inplace=True))

        self.conv4 = ConvModule(
            in_channels=in_ch,
            out_channels=out_ch,
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
        return input_high + low.mul(weight)


class RAUNet(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        # self.w = 512
        # self.h = 640
        filters = in_channels
        # resnet = models.resnet34(pretrained=pretrained)
        # filters = [256, 512, 1024, 2048]
        # resnet = models.resnet50(pretrained=pretrained)
        # if in_ch==4:
        #     self.firstconv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # else:
        #     self.firstconv = resnet.conv1
        # self.firstbn = resnet.bn1
        # self.firstrelu = resnet.relu
        # self.firstmaxpool = resnet.maxpool
        # self.encoder1 = resnet.layer1
        # self.encoder2 = resnet.layer2
        # self.encoder3 = resnet.layer3
        # self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3],
                                            filters[2],
                                            norm_cfg=norm_cfg,
                                            act_cfg=act_cfg)
        self.decoder3 = DecoderBlockLinkNet(filters[2],
                                            filters[1], norm_cfg=norm_cfg,
                                            act_cfg=act_cfg)
        self.decoder2 = DecoderBlockLinkNet(filters[1],
                                            filters[0],
                                            norm_cfg=norm_cfg,
                                            act_cfg=act_cfg)
        self.decoder1 = DecoderBlockLinkNet(filters[0],
                                            filters[0],
                                            norm_cfg=norm_cfg,
                                            act_cfg=act_cfg)
        self.gau3 = AAM(filters[2], filters[2],
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg)  # RAUNet
        self.gau2 = AAM(filters[1], filters[1],
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg)
        self.gau1 = AAM(filters[0], filters[0],
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg)

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 2, stride=2)
        # self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalrelu1 = build_activation_layer(act_cfg)
        # self.finalconv2 = nn.Conv2d(32, 32, 3)
        # self.finalrelu2 = nn.ReLU(inplace=True)
        # self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # if x.shape[1] == 1:
        #     x = torch.cat([x, x, x], dim=1)  # 扩充为3通道
        # # Encoder
        # x = self.firstconv(x)
        # x = self.firstbn(x)
        # x = self.firstrelu(x)
        # low = x
        # x = self.firstmaxpool(x)
        # e1 = self.encoder1(x)
        # e2 = self.encoder2(e1)
        # e3 = self.encoder3(e2)
        # e4 = self.encoder4(e3)
        e1, e2, e3, e4 = x
        # high = e2
        d4 = self.decoder4(e4)
        b4 = self.gau3(d4, e3)
        d3 = self.decoder3(b4)
        b3 = self.gau2(d3, e2)
        d2 = self.decoder2(b3)
        b2 = self.gau1(d2, e1)
        d1 = self.decoder1(b2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        # f3 = self.finalconv2(f2)
        # f4 = self.finalrelu2(f3)
        # f5 = self.finalconv3(f4)

        # if self.num_classes > 1:
        #     x_out = F.log_softmax(f5, dim=1)
        # else:
        #     x_out = f5

        return [f2]


class DecoderBlockLinkNet(nn.Module):
    def __init__(self, in_channels, n_filters, norm_cfg, act_cfg):
        super().__init__()

        # self.relu = nn.ReLU(inplace=True)
        self.act = build_activation_layer(act_cfg)
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        # self.norm1 = nn.BatchNorm2d(in_channels // 4)
        _, self.norm1 = build_norm_layer(norm_cfg, in_channels // 4)

        # B, C/4, H, W -> B, C/4, 2 * H, 2 * W
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=4,
                                          stride=2, padding=1, output_padding=0)
        # self.norm2 = nn.BatchNorm2d(in_channels // 4)
        _, self.norm2 = build_norm_layer(norm_cfg, in_channels // 4)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        # self.norm3 = nn.BatchNorm2d(n_filters)
        _, self.norm3 = build_norm_layer(norm_cfg, n_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act(x)
        return x
