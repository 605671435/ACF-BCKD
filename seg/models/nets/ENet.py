##################################################################
# Reproducing the paper                                          #
# ENet - Real Time Semantic Segmentation                         #
# Paper: https://arxiv.org/pdf/1606.02147.pdf                    #
#                                                                #
# Copyright (c) 2019                                             #
# Authors: @iArunava <iarunavaofficial@gmail.com>                #
#          @AvivSham <mista2311@gmail.com>                       #
#                                                                #
# License: BSD License 3.0                                       #
#                                                                #
# The Code in this file is distributed for free                  #
# usage and modification with proper credits                     #
# directing back to this repository.                             #
##################################################################

import torch
import torch.nn as nn


class ENet(nn.Module):
    def __init__(self, in_channels, C):
        super().__init__()

        # Define class variables
        self.C = C

        # The initial block
        self.init = InitialBlock(in_channels=in_channels, out_channels=16 - in_channels)

        # The first bottleneck
        self.b10 = RDDNeck(dilation=1,
                           in_channels=16,
                           out_channels=64,
                           down_flag=True,
                           p=0.01)

        self.b11 = RDDNeck(dilation=1,
                           in_channels=64,
                           out_channels=64,
                           down_flag=False,
                           p=0.01)

        self.b12 = RDDNeck(dilation=1,
                           in_channels=64,
                           out_channels=64,
                           down_flag=False,
                           p=0.01)

        self.b13 = RDDNeck(dilation=1,
                           in_channels=64,
                           out_channels=64,
                           down_flag=False,
                           p=0.01)

        self.b14 = RDDNeck(dilation=1,
                           in_channels=64,
                           out_channels=64,
                           down_flag=False,
                           p=0.01)

        # The second bottleneck
        self.b20 = RDDNeck(dilation=1,
                           in_channels=64,
                           out_channels=128,
                           down_flag=True)

        self.b21 = RDDNeck(dilation=1,
                           in_channels=128,
                           out_channels=128,
                           down_flag=False)

        self.b22 = RDDNeck(dilation=2,
                           in_channels=128,
                           out_channels=128,
                           down_flag=False)

        self.b23 = ASNeck(in_channels=128,
                          out_channels=128)

        self.b24 = RDDNeck(dilation=4,
                           in_channels=128,
                           out_channels=128,
                           down_flag=False)

        self.b25 = RDDNeck(dilation=1,
                           in_channels=128,
                           out_channels=128,
                           down_flag=False)

        self.b26 = RDDNeck(dilation=8,
                           in_channels=128,
                           out_channels=128,
                           down_flag=False)

        self.b27 = ASNeck(in_channels=128,
                          out_channels=128)

        self.b28 = RDDNeck(dilation=16,
                           in_channels=128,
                           out_channels=128,
                           down_flag=False)

        # The third bottleneck
        self.b31 = RDDNeck(dilation=1,
                           in_channels=128,
                           out_channels=128,
                           down_flag=False)

        self.b32 = RDDNeck(dilation=2,
                           in_channels=128,
                           out_channels=128,
                           down_flag=False)

        self.b33 = ASNeck(in_channels=128,
                          out_channels=128)

        self.b34 = RDDNeck(dilation=4,
                           in_channels=128,
                           out_channels=128,
                           down_flag=False)

        self.b35 = RDDNeck(dilation=1,
                           in_channels=128,
                           out_channels=128,
                           down_flag=False)

        self.b36 = RDDNeck(dilation=8,
                           in_channels=128,
                           out_channels=128,
                           down_flag=False)

        self.b37 = ASNeck(in_channels=128,
                          out_channels=128)

        self.b38 = RDDNeck(dilation=16,
                           in_channels=128,
                           out_channels=128,
                           down_flag=False)

        # The fourth bottleneck
        self.b40 = UBNeck(in_channels=128,
                          out_channels=64,
                          relu=True)

        self.b41 = RDDNeck(dilation=1,
                           in_channels=64,
                           out_channels=64,
                           down_flag=False,
                           relu=True)

        self.b42 = RDDNeck(dilation=1,
                           in_channels=64,
                           out_channels=64,
                           down_flag=False,
                           relu=True)

        # The fifth bottleneck
        self.b50 = UBNeck(in_channels=64,
                          out_channels=16,
                          relu=True)

        self.b51 = RDDNeck(dilation=1,
                           in_channels=16,
                           out_channels=16,
                           down_flag=False,
                           relu=True)

        # Final ConvTranspose Layer
        self.fullconv = nn.ConvTranspose3d(in_channels=16,
                                           out_channels=self.C,
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           output_padding=1,
                                           bias=False)

    def forward(self, x):
        # The initial block
        x = self.init(x)

        # The first bottleneck
        x, i1 = self.b10(x)
        x = self.b11(x)
        x = self.b12(x)
        x = self.b13(x)
        x = self.b14(x)

        # The second bottleneck
        x, i2 = self.b20(x)
        x = self.b21(x)
        x = self.b22(x)
        x = self.b23(x)
        x = self.b24(x)
        x = self.b25(x)
        x = self.b26(x)
        x = self.b27(x)
        x = self.b28(x)

        # The third bottleneck
        x = self.b31(x)
        x = self.b32(x)
        x = self.b33(x)
        x = self.b34(x)
        x = self.b35(x)
        x = self.b36(x)
        x = self.b37(x)
        x = self.b38(x)

        # The fourth bottleneck
        x = self.b40(x, i2)
        x = self.b41(x)
        x = self.b42(x)

        # The fifth bottleneck
        x = self.b50(x, i1)
        x = self.b51(x)

        # Final ConvTranspose Layer
        x = self.fullconv(x)

        return x


class InitialBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=13):
        super().__init__()

        self.maxpool = nn.MaxPool3d(kernel_size=2,
                                    stride=2,
                                    padding=0)

        self.conv = nn.Conv3d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)

        self.prelu = nn.PReLU(in_channels + out_channels)

        self.batchnorm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        main = self.conv(x)
        main = self.batchnorm(main)

        side = self.maxpool(x)

        x = torch.cat((main, side), dim=1)
        x = self.prelu(x)

        return x


class RDDNeck(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, down_flag, relu=False, projection_ratio=4, p=0.1):

        super().__init__()

        # Define class variables
        self.in_channels = in_channels

        self.out_channels = out_channels
        self.dilation = 1
        # self.dilation = dilation
        self.down_flag = down_flag

        if down_flag:
            self.stride = 2
            self.reduced_depth = int(in_channels // projection_ratio)
        else:
            self.stride = 1
            self.reduced_depth = int(out_channels // projection_ratio)

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        self.maxpool = nn.MaxPool3d(kernel_size=2,
                                    stride=2,
                                    padding=0, return_indices=True)

        self.dropout = nn.Dropout3d(p=0)
        # self.dropout = nn.Dropout3d(p=p)

        self.conv1 = nn.Conv3d(in_channels=self.in_channels,
                               out_channels=self.reduced_depth,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False,
                               dilation=1)

        self.prelu1 = activation

        self.conv2 = nn.Conv3d(in_channels=self.reduced_depth,
                               out_channels=self.reduced_depth,
                               kernel_size=3,
                               stride=self.stride,
                               padding=self.dilation,
                               bias=True,
                               dilation=self.dilation)

        self.prelu2 = activation

        self.conv3 = nn.Conv3d(in_channels=self.reduced_depth,
                               out_channels=self.out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False,
                               dilation=1)

        self.prelu3 = activation

        self.batchnorm = nn.BatchNorm3d(self.reduced_depth)
        self.batchnorm2 = nn.BatchNorm3d(self.out_channels)

    def forward(self, x):

        bs = x.size()[0]
        x_copy = x

        # Side Branch
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.prelu1(x)

        x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.prelu2(x)

        x = self.conv3(x)
        x = self.batchnorm2(x)

        x = self.dropout(x)

        # Main Branch
        if self.down_flag:
            x_copy, indices = self.maxpool(x_copy)

        if self.in_channels != self.out_channels:
            out_shape = self.out_channels - self.in_channels
            extras = torch.zeros((bs, out_shape, x.shape[2], x.shape[3], x.shape[4]))
            extras = extras.to(x_copy.device)
            x_copy = torch.cat((x_copy, extras), dim=1)

        # Sum of main and side branches
        x = x + x_copy
        x = self.prelu3(x)

        if self.down_flag:
            return x, indices
        else:
            return x


class UBNeck(nn.Module):
    def __init__(self, in_channels, out_channels, relu=False, projection_ratio=4):

        super().__init__()

        # Define class variables
        self.in_channels = in_channels
        self.reduced_depth = int(in_channels / projection_ratio)
        self.out_channels = out_channels

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        self.unpool = nn.MaxUnpool3d(kernel_size=2,
                                     stride=2)

        self.main_conv = nn.Conv3d(in_channels=self.in_channels,
                                   out_channels=self.out_channels,
                                   kernel_size=1)

        self.dropout = nn.Dropout3d(p=0)
        # self.dropout = nn.Dropout3d(p=0.1)

        self.convt1 = nn.ConvTranspose3d(in_channels=self.in_channels,
                                         out_channels=self.reduced_depth,
                                         kernel_size=1,
                                         padding=0,
                                         bias=False)

        self.prelu1 = activation

        self.convt2 = nn.ConvTranspose3d(in_channels=self.reduced_depth,
                                         out_channels=self.reduced_depth,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         output_padding=1,
                                         bias=False)

        self.prelu2 = activation

        self.convt3 = nn.ConvTranspose3d(in_channels=self.reduced_depth,
                                         out_channels=self.out_channels,
                                         kernel_size=1,
                                         padding=0,
                                         bias=False)

        self.prelu3 = activation

        self.batchnorm = nn.BatchNorm3d(self.reduced_depth)
        self.batchnorm2 = nn.BatchNorm3d(self.out_channels)

    def forward(self, x, indices):
        x_copy = x

        # Side Branch
        x = self.convt1(x)
        x = self.batchnorm(x)
        x = self.prelu1(x)

        x = self.convt2(x)
        x = self.batchnorm(x)
        x = self.prelu2(x)

        x = self.convt3(x)
        x = self.batchnorm2(x)

        x = self.dropout(x)

        # Main Branch

        x_copy = self.main_conv(x_copy)
        x_copy = self.unpool(x_copy, indices, output_size=x.size())

        # Concat
        x = x + x_copy
        x = self.prelu3(x)

        return x


class ASNeck(nn.Module):
    def __init__(self, in_channels, out_channels, projection_ratio=4):

        super().__init__()

        # Define class variables
        self.in_channels = in_channels
        self.reduced_depth = int(in_channels / projection_ratio)
        self.out_channels = out_channels

        self.dropout = nn.Dropout3d(p=0)
        # self.dropout = nn.Dropout3d(p=0.1)

        self.conv1 = nn.Conv3d(in_channels=self.in_channels,
                               out_channels=self.reduced_depth,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)

        self.prelu1 = nn.PReLU()

        self.conv21 = nn.Conv3d(in_channels=self.reduced_depth,
                                out_channels=self.reduced_depth,
                                kernel_size=(1, 5, 5),
                                stride=1,
                                padding=(0, 2, 2),
                                bias=False)

        self.conv22 = nn.Conv3d(in_channels=self.reduced_depth,
                                out_channels=self.reduced_depth,
                                kernel_size=(5, 1, 1),
                                stride=1,
                                padding=(2, 0, 0),
                                bias=False)

        self.prelu2 = nn.PReLU()

        self.conv3 = nn.Conv3d(in_channels=self.reduced_depth,
                               out_channels=self.out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)

        self.prelu3 = nn.PReLU()

        self.batchnorm = nn.BatchNorm3d(self.reduced_depth)
        self.batchnorm2 = nn.BatchNorm3d(self.out_channels)

    def forward(self, x):
        bs = x.size()[0]
        x_copy = x

        # Side Branch
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.prelu1(x)

        x = self.conv21(x)
        x = self.conv22(x)
        x = self.batchnorm(x)
        x = self.prelu2(x)

        x = self.conv3(x)

        x = self.dropout(x)
        x = self.batchnorm2(x)

        # Main Branch

        if self.in_channels != self.out_channels:
            out_shape = self.out_channels - self.in_channels
            extras = torch.zeros((bs, out_shape, x.shape[2], x.shape[3]))
            if torch.cuda.is_available():
                extras = extras.cuda()
            x_copy = torch.cat((x_copy, extras), dim=1)

        # Sum of main and side branches
        x = x + x_copy
        x = self.prelu3(x)

        return x


if __name__ == "__main__":
    x = torch.rand((1, 1, 96, 96, 96))

    model = ENet(1, 14)
    out = model(x)
    print(out.size())