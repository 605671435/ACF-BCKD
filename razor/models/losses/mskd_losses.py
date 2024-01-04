import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.autograd import Variable
from mmseg.models.utils import SELayer
from mmrazor.models.losses import L2Loss

# Feature Mapping Module (FMM)
class FMM(nn.Module):

    def __init__(self):
        super(FMM, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: Tensor):
        x = F.relu(x, inplace=True)
        channel_weights = self.global_avgpool(x)
        channel_weights = F.softmax(channel_weights, dim=1)
        out = channel_weights * x
        out = torch.mean(out, dim=1)
        return out


class FFD(nn.Module):
    def __init__(self, threshold=0.5, gamma=0.6):
        super(FFD, self).__init__()
        self.threshold = threshold
        self.gamma = gamma

    def forward(self, x):
        pf_x = F.sigmoid(x)
        pb_x = 1 - pf_x

        pf_x = (pf_x >= self.threshold) * pf_x
        pb_x = (pb_x >= self.threshold) * pb_x

        # pu_x: feature filtering map
        pu_x = pf_x + pb_x
        pu_x = pu_x.exp(self.gamma) / pu_x.mean()
        return pu_x


class FeatureFilteringLoss(nn.Module):

    def __init__(self, threshold=0.5, gamma=0.6, loss_weight=1.0):
        super(FeatureFilteringLoss, self).__init__()
        self.l2_loss = L2Loss(dist=True)
        self.ffm = FMM()
        self.ffd = FFD(threshold=threshold, gamma=gamma)
        self.loss_weight = loss_weight

    def forward(self, feats_S, feats_T):
        mapped_feats_S = self.ffm(feats_S)
        mapped_feats_T = self.ffm(feats_T)
        l2_dist = self.l2_loss(mapped_feats_S, mapped_feats_T)
        ffd_T = self.ffd(mapped_feats_T)
        loss = torch.mean(ffd_T * l2_dist, dim=1)
        return self.loss_weight * loss


class RegionGraphLoss(nn.Module):

    def __init__(self, patch_size=2, gamma=0.6, loss_weight=1.0):
        super(RegionGraphLoss, self).__init__()
        self.patch_size = patch_size
        self.l2_loss = L2Loss(dist=True)
        self.loss_weight = loss_weight

    def forward(self, feats_S, feats_T):
        B, C, H, W = feats_S.shape
        patches_S = F.adaptive_avg_pool2d(feats_S, output_size=(H // self.patch_size, W // self.patch_size))
        patches_T = F.adaptive_avg_pool2d(feats_T, output_size=(H // self.patch_size, W // self.patch_size))
        l2_dist = self.l2_loss(patches_T, patches_S)
        l_nodes = l2_dist.sum() / (H // self.patch_size)
        mapped_feats_S = self.ffm(feats_S)
        mapped_feats_T = self.ffm(feats_T)
        l2_dist = self.l2_loss(mapped_feats_S, mapped_feats_T)
        ffd_T = self.ffd(mapped_feats_T)
        loss = torch.mean(ffd_T * l2_dist, dim=1)
        return self.loss_weight * loss