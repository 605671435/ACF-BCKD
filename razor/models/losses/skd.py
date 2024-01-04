# copyright https://github.com/irfanICMLL/structure_knowledge_distillation/blob/master/utils/criterion.py#L228
import torch.nn as nn
import torch
from torch.nn import functional as F


def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0], 1, f_.shape[2], f_.shape[3], f_.shape[4]) + 1e-8


def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0], feat.shape[1], -1)
    return torch.einsum('icm,icn->imn', [feat, feat])


def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2]*f_T.shape[-3])**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis


class CriterionPixelWise(nn.Module):
    def __init__(self, ignore_index=255, loss_weight=1.0, reduce=True, sigmoid=False):
        super(CriterionPixelWise, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            print("disabled the reduce.")
        self.loss_weight = loss_weight
        self.sigmoid = sigmoid

    def forward(self, preds_S, preds_T):
        preds_T.detach()
        assert preds_S.shape == preds_T.shape, 'the output dim of teacher and student differ'
        N, C, W, H, D = preds_S.shape
        if self.sigmoid:
            sigmoid_pred_T = torch.sigmoid(preds_T)
            loss = (torch.sum(sigmoid_pred_T * torch.sigmoid(preds_S)))/(C*W*H*D)
        else:
            softmax_pred_T = F.softmax(preds_T.permute(0, 2, 3, 4, 1).contiguous().view(-1, C), dim=1)
            logsoftmax = nn.LogSoftmax(dim=1)
            loss = (torch.sum(
                - softmax_pred_T * logsoftmax(preds_S.permute(0, 2, 3, 4, 1).contiguous().view(-1, C))))/(W*H*D)
        return self.loss_weight * loss


class CriterionPairWiseforWholeFeatAfterPool(nn.Module):
    def __init__(self, scale=0.5, loss_weight=1.0, sigmoid=False):
        """inter pair-wise loss from inter feature maps"""
        super(CriterionPairWiseforWholeFeatAfterPool, self).__init__()
        self.criterion = sim_dis_compute
        self.scale = scale
        self.loss_weight = loss_weight
        self.sigmoid = sigmoid

    def forward(self, feat_S, feat_T):
        feat_T.detach()

        total_w, total_h, total_d = feat_T.shape[2], feat_T.shape[3], feat_T.shape[4]
        patch_w, patch_h, patch_d = int(total_w*self.scale), int(total_h*self.scale), int(total_d*self.scale)
        maxpool = nn.MaxPool3d(
            kernel_size=(patch_w, patch_h, patch_d),
            stride=(patch_w, patch_h, patch_d),
            padding=0,
            ceil_mode=True) # change
        loss = self.criterion(maxpool(feat_S), maxpool(feat_T))
        return self.loss_weight * loss
