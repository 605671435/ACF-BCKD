# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F


class DKDLoss(nn.Module):
    """Decoupled Knowledge Distillation, CVPR2022.

    link: https://arxiv.org/abs/2203.08679
    reformulate the classical KD loss into two parts:
        1. target class knowledge distillation (TCKD)
        2. non-target class knowledge distillation (NCKD).
    Args:
    tau (float): Temperature coefficient. Defaults to 1.0.
    alpha (float): Weight of TCKD loss. Defaults to 1.0.
    beta (float): Weight of NCKD loss. Defaults to 1.0.
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
    """

    def __init__(
        self,
        tau: float = 1.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        reduction: str = 'batchmean',
        loss_weight: float = 1.0,
    ) -> None:
        super(DKDLoss, self).__init__()
        self.tau = tau
        accept_reduction = {'none', 'batchmean', 'sum', 'mean'}
        assert reduction in accept_reduction, \
            f'KLDivergence supports reduction {accept_reduction}, ' \
            f'but gets {reduction}.'
        self.reduction = reduction
        self.alpha = alpha
        self.beta = beta
        self.loss_weight = loss_weight

    def forward(
        self,
        preds_S: torch.Tensor,
        preds_T: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> torch.Tensor:
        """DKDLoss forward function.

        Args:
            preds_S (torch.Tensor): The student model prediction, shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction, shape (N, C).
            gt_labels (torch.Tensor): The gt label tensor, shape (N, ).

        Return:
            torch.Tensor: The calculated loss value.
        """
        C = preds_S.shape[1]
        preds_S = preds_S.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        preds_T = preds_T.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        gt_labels = gt_labels.contiguous().view(-1).long()
        gt_mask = self._get_gt_mask(preds_S, gt_labels)
        tckd_loss = self._get_tckd_loss(preds_S, preds_T, gt_labels, gt_mask)
        nckd_loss = self._get_nckd_loss(preds_S, preds_T, gt_mask)
        loss = self.alpha * tckd_loss + self.beta * nckd_loss
        return self.loss_weight * loss

    def _get_nckd_loss(
        self,
        preds_S: torch.Tensor,
        preds_T: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate non-target class knowledge distillation."""
        # implementation to mask out gt_mask, faster than index
        s_nckd = F.log_softmax(preds_S / self.tau - 1000.0 * gt_mask, dim=1)
        t_nckd = F.softmax(preds_T / self.tau - 1000.0 * gt_mask, dim=1)
        return self._kl_loss(s_nckd, t_nckd)

    def _get_tckd_loss(
        self,
        preds_S: torch.Tensor,
        preds_T: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate target class knowledge distillation."""
        non_gt_mask = self._get_non_gt_mask(preds_S, gt_labels)
        s_tckd = F.softmax(preds_S / self.tau, dim=1)
        t_tckd = F.softmax(preds_T / self.tau, dim=1)
        mask_student = torch.log(self._cat_mask(s_tckd, gt_mask, non_gt_mask))
        mask_teacher = self._cat_mask(t_tckd, gt_mask, non_gt_mask)
        return self._kl_loss(mask_student, mask_teacher)

    def _kl_loss(
        self,
        preds_S: torch.Tensor,
        preds_T: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the KL Divergence."""
        kl_loss = F.kl_div(
            preds_S, preds_T, size_average=False,
            reduction=self.reduction) * self.tau**2
        return kl_loss

    def _cat_mask(
        self,
        tckd: torch.Tensor,
        gt_mask: torch.Tensor,
        non_gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate preds of target (pt) & preds of non-target (pnt)."""
        t1 = (tckd * gt_mask).sum(dim=1, keepdims=True)
        t2 = (tckd * non_gt_mask).sum(dim=1, keepdims=True)
        return torch.cat([t1, t2], dim=1)

    def _get_gt_mask(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate groundtruth mask on logits with target class tensor.

        Args:
            logits (torch.Tensor): The prediction logits with shape (N, C).
            target (torch.Tensor): The gt_label target with shape (N, C).

        Return:
            torch.Tensor: The masked logits.
        """
        target = target.reshape(-1)
        return torch.zeros_like(logits).scatter_(1, target.unsqueeze(1),
                                                 1).bool()

    def _get_non_gt_mask(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate non-groundtruth mask on logits with target class tensor.

        Args:
            logits (torch.Tensor): The prediction logits with shape (N, C).
            target (torch.Tensor): The gt_label target with shape (N, C).

        Return:
            torch.Tensor: The masked logits.
        """
        target = target.reshape(-1)
        return torch.ones_like(logits).scatter_(1, target.unsqueeze(1),
                                                0).bool()


class DKDLossBraTS23(DKDLoss):
    #
    # def __init__(self, **kwargs) -> None:
    #     super(DKDLossBraTS23, self).__init__(**kwargs)

    # def forward_per_class(
    def forward(
        self,
        preds_S: torch.Tensor,
        preds_T: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> torch.Tensor:
        """DKDLoss forward function.

        Args:
            preds_S (torch.Tensor): The student model prediction, shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction, shape (N, C).
            gt_labels (torch.Tensor): The gt label tensor, shape (N, ).

        Return:
            torch.Tensor: The calculated loss value.
        """
        B, C, H, W, D = preds_S.shape
        preds_S = preds_S
        preds_T = preds_T
        # gt_labels = gt_labels.contiguous().view(-1).unsqueeze(-1)
        gt_labels = gt_labels
        gt_mask = gt_labels
        tckd_loss = self._get_tckd_loss(preds_S, preds_T, gt_labels, gt_mask) / (B * C * H * W * D)
        nckd_loss = self._get_nckd_loss(preds_S, preds_T, gt_mask) / (B * C * H * W * D)
        loss = self.alpha * tckd_loss + self.beta * nckd_loss
        return self.loss_weight * loss

    def _kl_loss(
        self,
        preds_S: torch.Tensor,
        preds_T: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the KL Divergence."""
        kl_loss = F.mse_loss(
            preds_S, preds_T, size_average=False,
            reduction=self.reduction)
        return kl_loss

    def _get_tckd_loss(
        self,
        preds_S: torch.Tensor,
        preds_T: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate target class knowledge distillation."""
        non_gt_mask = ~gt_mask
        s_tckd = torch.sigmoid(preds_S)
        t_tckd = torch.sigmoid(preds_T)
        mask_student = torch.cat([s_tckd * gt_mask, s_tckd * non_gt_mask], dim=1)
        mask_teacher = torch.cat([t_tckd * gt_mask, t_tckd * non_gt_mask], dim=1)
        return self._kl_loss(mask_student, mask_teacher)

    def _get_nckd_loss(
        self,
        preds_S: torch.Tensor,
        preds_T: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate non-target class knowledge distillation."""
        # implementation to mask out gt_mask, faster than index
        s_nckd = torch.sigmoid(preds_S * ~gt_mask)
        t_nckd = torch.sigmoid(preds_T * ~gt_mask)
        return self._kl_loss(s_nckd, t_nckd)

    # def forward(
    #     self,
    #     preds_S: torch.Tensor,
    #     preds_T: torch.Tensor,
    #     gt_labels: torch.Tensor,
    # ) -> torch.Tensor:
    #     loss = torch.tensor(0.).cuda()
    #     C = preds_S.shape[1]
    #     for c in range(C):
    #         preds_S_c = preds_S[:, c, ...].unsqueeze(1)
    #         preds_T_c = preds_T[:, c, ...].unsqueeze(1)
    #         gt_labels_c = gt_labels[:, c, ...]
    #         loss_c = self.forward_per_class(preds_S_c, preds_T_c, gt_labels_c)
    #         loss += loss_c
    #     loss = loss / C
    #     return loss
