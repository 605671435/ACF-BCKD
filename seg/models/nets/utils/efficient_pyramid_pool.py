#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================

import torch
from torch import nn
import math
from torch.nn import functional as F
from .cnn_utils import CBR, BR, Shuffle

class EfficientPyrPool(nn.Module):
    """Efficient Pyramid Pooling Module"""

    def __init__(self, in_planes, proj_planes, out_planes, scales=[2.0, 1.5, 1.0, 0.5, 0.1], last_layer_br=True):
        super(EfficientPyrPool, self).__init__()
        self.stages = nn.ModuleList()
        scales.sort(reverse=True)

        self.projection_layer = CBR(in_planes, proj_planes, 1, 1)
        for _ in enumerate(scales):
            self.stages.append(nn.Conv3d(proj_planes, proj_planes, kernel_size=3, stride=1, padding=1, bias=False, groups=proj_planes))

        self.merge_layer = nn.Sequential(
            # perform one big batch normalization instead of p small ones
            BR(proj_planes * len(scales)),
            Shuffle(groups=len(scales)),
            CBR(proj_planes * len(scales), proj_planes, 3, 1, groups=proj_planes),
            nn.Conv3d(proj_planes, out_planes, kernel_size=1, stride=1, bias=not last_layer_br),
        )
        if last_layer_br:
            self.br = BR(out_planes)
        self.last_layer_br = last_layer_br
        self.scales = scales

    def forward(self, x):
        hs = []
        x = self.projection_layer(x)
        height, width, dim = x.size()[2:]
        for i, stage in enumerate(self.stages):
            h_s = int(math.ceil(height * self.scales[i]))
            w_s = int(math.ceil(width * self.scales[i]))
            d_s = int(math.ceil(dim * self.scales[i]))
            h_s = h_s if h_s > 5 else 5
            w_s = w_s if w_s > 5 else 5
            d_s = d_s if d_s > 5 else 5
            if self.scales[i] < 1.0:
                h = F.adaptive_avg_pool3d(x, output_size=(h_s, w_s, d_s))
                h = stage(h)
                h = F.interpolate(h, (height, width, dim), mode='trilinear', align_corners=True)
            elif self.scales[i] > 1.0:
                h = F.interpolate(x, (h_s, w_s, d_s), mode='trilinear', align_corners=True)
                h = stage(h)
                h = F.adaptive_avg_pool3d(h, output_size=(height, width, dim))
            else:
                h = stage(x)
            hs.append(h)

        out = torch.cat(hs, dim=1)
        out = self.merge_layer(out)
        if self.last_layer_br:
            return self.br(out)
        return out