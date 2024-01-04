# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivergence(nn.Module):
    """A measure of how one probability distribution Q is different from a
    second, reference probability distribution P.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        reduction (str): Specifies the reduction to apply to the loss:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied,
            ``'batchmean'``: the sum of the output will be divided by
                the batchsize,
            ``'sum'``: the output will be summed,
            ``'mean'``: the output will be divided by the number of
                elements in the output.
            Default: ``'batchmean'``
        loss_weight (float): Weight of loss. Defaults to 1.0.
        teacher_detach (bool): Whether to detach the teacher model prediction.
            Will set to ``'False'`` in some data-free distillation algorithms.
            Defaults to True.
    """

    def __init__(
        self,
        tau: float = 1.0,
        reduction: str = 'batchmean',
        loss_weight: float = 1.0,
        teacher_detach: bool = True,
    ):
        super(KLDivergence, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight
        self.teacher_detach = teacher_detach

        accept_reduction = {'none', 'batchmean', 'sum', 'mean'}
        assert reduction in accept_reduction, \
            f'KLDivergence supports reduction {accept_reduction}, ' \
            f'but gets {reduction}.'
        self.reduction = reduction

    def forward(self, preds_S: torch.Tensor, preds_T: torch.Tensor):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).

        Return:
            torch.Tensor: The calculated loss value.
        """
        num_classes = preds_S.shape[1]
        if self.teacher_detach:
            preds_T = preds_T.detach()
        softmax_pred_T = F.softmax(preds_T / self.tau, dim=1).view(-1, num_classes)
        logsoftmax_preds_S = F.log_softmax(preds_S / self.tau, dim=1).view(-1, num_classes)

        loss = (self.tau**2) * F.kl_div(
            logsoftmax_preds_S, softmax_pred_T, reduction='batchmean')
        return self.loss_weight * loss


class KLDivergence2(nn.Module):
    """A measure of how one probability distribution Q is different from a
    second, reference probability distribution P.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        reduction (str): Specifies the reduction to apply to the loss:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied,
            ``'batchmean'``: the sum of the output will be divided by
                the batchsize,
            ``'sum'``: the output will be summed,
            ``'mean'``: the output will be divided by the number of
                elements in the output.
            Default: ``'batchmean'``
        loss_weight (float): Weight of loss. Defaults to 1.0.
        teacher_detach (bool): Whether to detach the teacher model prediction.
            Will set to ``'False'`` in some data-free distillation algorithms.
            Defaults to True.
    """

    def __init__(
        self,
        tau: float = 1.0,
        reduction: str = 'batchmean',
        loss_weight: float = 1.0,
        teacher_detach: bool = True,
    ):
        super(KLDivergence2, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight
        self.teacher_detach = teacher_detach

        accept_reduction = {'none', 'batchmean', 'sum', 'mean'}
        assert reduction in accept_reduction, \
            f'KLDivergence supports reduction {accept_reduction}, ' \
            f'but gets {reduction}.'
        self.reduction = reduction

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).

        Return:
            torch.Tensor: The calculated loss value.
        """
        if self.teacher_detach:
            preds_T = preds_T.detach()
        softmax_pred_T = F.softmax(preds_T / self.tau, dim=1)
        logsoftmax_preds_S = F.log_softmax(preds_S / self.tau, dim=1)

        loss = (self.tau**2) * F.kl_div(
            logsoftmax_preds_S, softmax_pred_T, reduction='none', log_target=False)

        batch_size = preds_S.shape[0]

        if self.reduction == 'sum':
            # Change view to calculate instance-wise sum
            loss = loss.view(batch_size, -1)
            return self.loss_weight * torch.sum(loss, dim=1)

        elif self.reduction == 'mean':
            # Change view to calculate instance-wise mean
            loss = loss.view(batch_size, -1)
            return self.loss_weight * torch.mean(loss, dim=1)


class KLDivergence3(nn.Module):
    def __init__(
        self,
        tau: float = 1.0,
        reduction: str = 'batchmean',
        loss_weight: float = 1.0,
        teacher_detach: bool = True,
    ):
        super(KLDivergence3, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight
        self.teacher_detach = teacher_detach

        accept_reduction = {'none', 'batchmean', 'sum', 'mean'}
        assert reduction in accept_reduction, \
            f'KLDivergence supports reduction {accept_reduction}, ' \
            f'but gets {reduction}.'
        self.reduction = reduction

    def forward(self, preds_S, preds_T):
        if self.teacher_detach:
            preds_T = preds_T.detach()
        C = preds_S.shape[1]
        softmax_pred_T = F.softmax(preds_T / self.tau, dim=1).permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        logsoftmax_preds_S = F.log_softmax(preds_S / self.tau, dim=1).permute(0, 2, 3, 4, 1).contiguous().view(-1, C)

        loss = (self.tau**2) * F.kl_div(
            logsoftmax_preds_S, softmax_pred_T, reduction='mean')

        return self.loss_weight * loss


class CriterionKD(nn.Module):
    '''
    knowledge distillation loss
    '''

    def __init__(self,
                 tau: float = 1.0,
                 sigmoid: bool = True,
                 loss_weight: float = 1.0):
        super(CriterionKD, self).__init__()
        self.temperature = tau
        self.sigmoid = sigmoid
        self.loss_weight = loss_weight
        if self.sigmoid:
            self.criterion_kd = torch.nn.MSELoss()
        else:
            self.criterion_kd = torch.nn.KLDivLoss()

    def forward(self, preds_S, preds_T):
        preds_T.detach()
        if self.sigmoid:
            loss = self.criterion_kd(
                torch.sigmoid(preds_S),
                torch.sigmoid(preds_T))
        else:
            loss = self.criterion_kd(
                F.log_softmax(preds_S / self.temperature, dim=1),
                F.softmax(preds_T / self.temperature, dim=1))
        return self.loss_weight * loss

# Implement in CIRKD:
# class CriterionKD(nn.Module):
#     '''
#     knowledge distillation loss
#     '''
#     def __init__(self, temperature=1):
#         super(CriterionKD, self).__init__()
#         self.temperature = temperature
#
#     def forward(self, pred, soft):
#         B, C, h, w = soft.size()
#         scale_pred = pred.permute(0,2,3,1).contiguous().view(-1,C)
#         scale_soft = soft.permute(0,2,3,1).contiguous().view(-1,C)
#         p_s = F.log_softmax(scale_pred / self.temperature, dim=1)
#         p_t = F.softmax(scale_soft / self.temperature, dim=1)
#         loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.temperature**2)
#         return loss