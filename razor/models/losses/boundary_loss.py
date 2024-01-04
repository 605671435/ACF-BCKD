# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.losses.huasdorff_distance_loss import HuasdorffDisstanceLoss
# from mmseg.models.decode_heads import PIDHead
# from mmseg.models.backbones import PIDNet
# from mmseg.datasets.transforms import GenerateEdge
def L2(f_):
    return (((f_**2).sum())**0.5) + 1e-8


def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    return torch.einsum('m,n->mn', [feat, feat])


def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2) / f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis


def boundary_pattern():
    matrix = torch.ones((3, 3, 3), dtype=torch.float32, device='cpu')
    matrix[1:-1, 1:-1, 1:-1] = 0
    matrix = matrix.view(1, 1, 3, 3, 3).cuda()
    return matrix

class BoundaryKDV1(nn.Module):
    '''
    knowledge distillation loss
    '''

    def __init__(self,
                 tau: float = 1.0,
                 num_classes: int = 14,
                 one_hot_target: bool = True,
                 include_background: bool = True,
                 loss_weight: float = 1.0):
        super(BoundaryKDV1, self).__init__()
        self.temperature = tau
        self.loss_weight = loss_weight
        self.kernel = boundary_pattern()
        self.num_classes = num_classes
        self.one_hot_target = one_hot_target
        self.include_background = include_background
        self.criterion_kd = torch.nn.KLDivLoss()

    def get_boundary(self, gt: torch.Tensor, cls: int) -> torch.Tensor:
        if self.one_hot_target:
            gt_cls = gt == cls
        else:
            gt_cls = gt[cls, ...].unsqueeze(0)
        boundary = F.conv3d(gt_cls.float(), self.kernel, padding=1)
        boundary[boundary == self.kernel.sum()] = 0
        boundary[boundary > 0] = 1
        return boundary

    def forward(self, preds_S, preds_T, gt_labels):
        batch_size, C, H, W, D = preds_S.shape
        loss = torch.tensor(0.).cuda()
        for bs in range(batch_size):
            preds_S_i = preds_S[bs].contiguous().view(preds_S.shape[1], -1)
            preds_T_i = preds_T[bs].contiguous().view(preds_T.shape[1], -1)
            preds_T_i.detach()
            for cls in range(self.num_classes):
                if cls == 0 and not self.include_background:
                    continue
                boundary = self.get_boundary(gt_labels[bs].detach().clone(), cls)
                boundary = boundary.view(-1)
                idxs = (boundary == 1).nonzero()
                if idxs.sum() == 0:
                    continue
                boundary_S = preds_S_i[:, idxs].squeeze(-1)
                boundary_T = preds_T_i[:, idxs].squeeze(-1)
                if self.one_hot_target:
                    loss += F.kl_div(
                        F.log_softmax(boundary_S / self.temperature, dim=0),
                        F.softmax(boundary_T / self.temperature, dim=0))
                else:
                    loss += F.mse_loss(
                        torch.sigmoid(boundary_S),
                        torch.sigmoid(boundary_T)
                    )

        return self.loss_weight * loss


class BoundaryKDV3(nn.Module):
    '''
    knowledge distillation loss
    '''

    def __init__(self,
                 tau: float = 1.0,
                 num_classes: int = 14,
                 one_hot_target: bool = True,
                 include_background: bool = True,
                 loss_weight: float = 1.0):
        super(BoundaryKDV3, self).__init__()
        self.temperature = tau
        self.loss_weight = loss_weight
        self.one_hot_target = one_hot_target
        self.kernel = boundary_pattern()
        self.num_classes = num_classes
        self.include_background = include_background
        self.criterion_kd = torch.nn.KLDivLoss()

    def get_boundary(self, gt: torch.Tensor, cls: int) -> torch.Tensor:
        if self.one_hot_target:
            gt_cls = gt == cls
        else:
            gt_cls = gt[cls, ...].unsqueeze(0)
        boundary = F.conv3d(gt_cls.float(), self.kernel, padding=1)
        boundary[boundary == self.kernel.sum()] = 0
        boundary[boundary > 0] = 1
        return boundary

    def forward(self, preds_S, preds_T, gt_labels):
        batch_size, C, H, W, D = preds_S.shape
        # loss = torch.tensor(0.).cuda()
        all_loss = []
        for bs in range(batch_size):
            preds_S_i = preds_S[bs].contiguous().view(preds_S.shape[1], -1)
            preds_T_i = preds_T[bs].contiguous().view(preds_T.shape[1], -1)
            preds_T_i.detach()
            for cls in range(self.num_classes):
                if cls == 0 and not self.include_background:
                    continue
                boundary = self.get_boundary(gt_labels[bs].detach().clone(), cls)
                boundary = boundary.view(-1)
                idxs = (boundary == 1).nonzero()
                if idxs.sum() == 0:
                    continue
                boundary_S = preds_S_i[:, idxs].squeeze(-1)
                boundary_T = preds_T_i[:, idxs].squeeze(-1)
                if self.one_hot_target:
                    l = F.kl_div(
                        F.log_softmax(boundary_S / self.temperature, dim=0),
                        F.softmax(boundary_T / self.temperature, dim=0))
                else:
                    l = F.mse_loss(
                        torch.sigmoid(boundary_S),
                        torch.sigmoid(boundary_T)
                    )
                all_loss.append(l.unsqueeze(0))
        if len(all_loss) == 0:
            return torch.tensor(0.).cuda()
        loss = torch.cat(all_loss, dim=0)
        loss = torch.mean(loss)
        return self.loss_weight * loss

class BoundaryKD(nn.Module):
    '''
    knowledge distillation loss
    '''

    def __init__(self,
                 tau: float = 1.0,
                 num_classes: int = 14,
                 loss_weight: float = 1.0):
        super(BoundaryKD, self).__init__()
        self.temperature = tau
        self.loss_weight = loss_weight
        self.kernel = boundary_pattern()
        self.num_classes = num_classes
        self.criterion_kd = torch.nn.KLDivLoss()

    def pair_wise_sim_map(self, fea_0, fea_1):
        C, H, W, D = fea_0.size()

        fea_0 = fea_0.reshape(C, -1).transpose(0, 1)
        fea_1 = fea_1.reshape(C, -1).transpose(0, 1)

        sim_map_0_1 = torch.mm(fea_0, fea_1.transpose(0, 1))
        return sim_map_0_1

    def get_boundary(self, gt: torch.Tensor, cls: int) -> torch.Tensor:
        gt_cls = gt == cls
        boundary = F.conv3d(gt_cls.float(), self.kernel, padding=1)
        boundary[boundary == self.kernel.sum()] = 0
        boundary[boundary > 0] = 1
        return boundary

    def sim_kd(self, s_logits, t_logits):
        p_s = F.log_softmax(s_logits / self.temperature, dim=1)
        p_t = F.softmax(t_logits / self.temperature, dim=1)
        sim_dis = F.kl_div(p_s, p_t, reduction='batchmean') * self.temperature ** 2
        return sim_dis

    def forward(self, preds_S, preds_T, gt_labels):
        batch_size, C, H, W, D = preds_S.shape
        preds_S = preds_S.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        preds_T = preds_T.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)

        loss = torch.tensor(0.).cuda()

        preds_T.detach()
        for cls in range(1, self.num_classes):
            boundary = self.get_boundary(gt_labels.detach().clone(), cls)
            boundary = boundary.view(-1)
            idxs = (boundary == 1).nonzero()
            if idxs.sum() == 0:
                continue
            # num_classes x num_pixels
            boundary_S = preds_S[idxs, :].squeeze(1)
            boundary_T = preds_T[idxs, :].squeeze(1)
            loss += F.kl_div(
                F.log_softmax(boundary_S / self.temperature, dim=1),
                F.softmax(boundary_T / self.temperature, dim=1))
        loss = loss / (self.num_classes - 1)
        return self.loss_weight * loss


class AreaKD(nn.Module):
    '''
    knowledge distillation loss
    '''

    def __init__(self,
                 tau: float = 1.0,
                 num_classes: int = 14,
                 loss_weight: float = 1.0):
        super(AreaKD, self).__init__()
        self.temperature = tau
        self.loss_weight = loss_weight
        self.kernel = boundary_pattern()
        self.num_classes = num_classes
        self.criterion_kd = torch.nn.KLDivLoss()

    def get_area(self, gt: torch.Tensor, cls: int) -> torch.Tensor:
        area = gt == cls
        area = area.float()
        return area

    def forward(self, preds_S, preds_T, gt_labels):
        batch_size, C, H, W, D = preds_S.shape
        # loss = torch.tensor(0.).cuda()
        preds_S = preds_S.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        preds_T = preds_T.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        gt_labels = gt_labels.view(-1, 1)
        onehot_mask = torch.zeros_like(preds_S).scatter_(1, gt_labels.long(), 1).bool()
        # preds_T.detach()

        loss = F.kl_div(
            F.log_softmax(preds_S * onehot_mask / self.temperature, dim=1),
            F.softmax(preds_T * onehot_mask / self.temperature, dim=1),
            size_average=False,
            reduction='batchmean') * self.temperature**2

        # for cls in range(1, self.num_classes):
        #     area = self.get_area(gt_labels.detach().clone(), cls)
        #     area = area.view(-1)
        #     idxs = (area == 1).nonzero()
        #     if idxs.sum() == 0:
        #         continue
        #     area_S = preds_S[idxs, cls].squeeze(-1)
        #     area_T = preds_T[idxs, cls].squeeze(-1)
        #     loss += F.kl_div(
        #         F.log_softmax(area_S / self.temperature, dim=0),
        #         F.softmax(area_T / self.temperature, dim=0))

        return self.loss_weight * loss

