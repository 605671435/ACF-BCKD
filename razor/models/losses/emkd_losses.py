import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.autograd import Variable


def prediction_map_distillation(y: Tensor, teacher_scores: Tensor, tau: int = 4):
    """
    basic KD loss function based on "Distilling the Knowledge in a Neural Network"
    https://arxiv.org/abs/1503.02531
    :param y: student score map
    :param teacher_scores: teacher score map
    :param T:  for softmax
    :return: loss value
    """
    num_classes = y.shape[1]
    p = F.log_softmax(y / tau, dim=1)
    q = F.softmax(teacher_scores / tau, dim=1)

    p = p.view(-1, num_classes)
    q = q.view(-1, num_classes)

    l_kl = F.kl_div(p, q, reduction='batchmean') * (tau ** 2)
    return l_kl


def at(x, exp):
    """
    attention value of a feature map
    :param x: feature
    :return: attention value
    """
    return F.normalize(x.pow(exp).mean(1).view(x.size(0), -1))


def importance_maps_distillation(s, t, exp=4):
    """
    importance_maps_distillation KD loss, based on "Paying More Attention to Attention:
    Improving the Performance of Convolutional Neural Networks via Attention Transfer"
    https://arxiv.org/abs/1612.03928
    :param exp: exponent
    :param s: student feature maps
    :param t: teacher feature maps
    :return: imd loss value
    """
    if s.shape[2:] != t.shape[2:]:
        s = F.interpolate(s, t.size()[-3:], mode='nearest')
    return torch.sum((at(s, exp) - at(t, exp)).pow(2), dim=1).mean()


class PMD(nn.Module):

    def __init__(self, loss_weight: float = 1.0, tau: int = 4):
        super(PMD, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight

    def forward(self, preds_S: Tensor, preds_T: Tensor):
        loss = prediction_map_distillation(preds_S, preds_T, self.tau)
        return self.loss_weight * loss

    # def forward(self, preds_S, preds_T):
    #     preds_T[0].detach()
    #     assert preds_S.shape == preds_T.shape, 'the output dim of teacher and student differ'
    #     N, C, W, H, D = preds_S.shape
    #     softmax_pred_T = F.softmax(preds_T / self.tau, dim=1).permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
    #     # logsoftmax = nn.LogSoftmax(dim=1)
    #     logsoftmax = F.log_softmax(preds_S / self.tau, dim=1).permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
    #     loss = (torch.sum(softmax_pred_T * logsoftmax)) / (W * H * D)
    #     return loss


class IMD(nn.Module):

    def __init__(self, exp=4, loss_weight=1.0):
        super(IMD, self).__init__()
        self.exp = exp
        self.loss_weight = loss_weight

    def forward(self, s_feature, t_feature):
        loss = importance_maps_distillation(s_feature, t_feature, self.exp)
        return self.loss_weight * loss


class RAD(nn.Module):

    def __init__(self, num_classes=9, loss_weight=1.0):
        super(RAD, self).__init__()
        self.num_classes = num_classes
        self.loss_weight = loss_weight

    def forward(self, logits_S, logits_T, ground_truth):
        loss = self.region_affinity_distillation(logits_S, logits_T, ground_truth)
        return self.loss_weight * loss

    def region_contrast(self, x, gt_i, gt_j):
        """
        calculate region contrast value
        :param x: feature
        :param gt: mask
        :return: value
        """
        smooth = 1.0

        mask0 = gt_i.unsqueeze(1)
        mask1 = gt_j.unsqueeze(1)

        region0 = torch.sum(x * mask0, dim=(2, 3, 4)) / torch.sum(mask0, dim=(2, 3, 4))
        region1 = torch.sum(x * mask1, dim=(2, 3, 4)) / (torch.sum(mask1, dim=(2, 3, 4)) + smooth)
        return F.cosine_similarity(region0, region1, dim=1)

    def region_affinity_distillation(self, s, t, gt):
        """
        region affinity distillation KD loss
        :param s: student feature
        :param t: teacher feature
        :return: loss value
        """
        gt = F.interpolate(gt.float(), s.size()[2:]).squeeze(1)
        # gt = F.one_hot(gt.squeeze(1)).float()
        gt = F.one_hot(
            torch.clamp(gt.long(), 0, self.num_classes - 1),
            num_classes=self.num_classes).float()

        loss = torch.Tensor([0.]).cuda()

        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                # if i == j:
                #     continue
                gt_i = gt[..., i]
                gt_j = gt[..., j]
                if gt_i.sum() != 0 and gt_j.sum() != 0:
                    loss_ij = (self.region_contrast(s, gt_i, gt_j)
                               - self.region_contrast(t, gt_i, gt_j)).pow(2).mean()
                    if not torch.isnan(loss_ij):
                        loss += loss_ij
                    # print(f'i={i}, j={j}, loss={loss_ij}')

        # return (self.region_contrast(s, gt) - self.region_contrast(t, gt)).pow(2).mean()
        return loss


class RAD_BraTS23(nn.Module):

    def __init__(self, num_classes=9, loss_weight=1.0):
        super(RAD_BraTS23, self).__init__()
        self.num_classes = num_classes
        self.loss_weight = loss_weight

    def forward(self, logits_S, logits_T, ground_truth):
        loss = self.region_affinity_distillation(logits_S, logits_T, ground_truth)
        return self.loss_weight * loss

    def region_contrast(self, x, gt):
        """
        calculate region contrast value
        :param x: feature
        :param gt: mask
        :return: value
        """
        smooth = 1.0
        mask0 = gt.bool()
        mask1 = ~mask0

        region0 = torch.sum(x * mask0, dim=(2, 3, 4)) / (torch.sum(mask1, dim=(2, 3, 4)) + smooth)
        region1 = torch.sum(x * mask1, dim=(2, 3, 4)) / (torch.sum(mask1, dim=(2, 3, 4)) + smooth)
        return F.cosine_similarity(region0, region1, dim=1)

    def region_affinity_distillation(self, s, t, gt):
        """
        region affinity distillation KD loss
        :param s: student feature
        :param t: teacher feature
        :return: loss value
        """
        gt = F.interpolate(gt.float(), s.size()[2:])
        loss = torch.tensor(0.).cuda()
        for i in range(self.num_classes):
            loss += (self.region_contrast(s, gt[:, [i], ...]) - self.region_contrast(t, gt[:, [i], ...])).pow(2).mean()
        loss = loss / self.num_classes
        return loss
