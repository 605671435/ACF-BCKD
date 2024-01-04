# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from mmrazor.models.losses import ChannelWiseDivergence as _CWD


class ChannelWiseDivergence(_CWD):

    def __init__(self, sigmoid: bool = False, **kwargs):
        super(ChannelWiseDivergence, self).__init__(**kwargs)
        self.sigmoid = sigmoid

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W, D).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W, D).

        Return:
            torch.Tensor: The calculated loss value.
        """
        assert preds_S.shape[-3:] == preds_T.shape[-3:]
        N, C, H, W, D = preds_S.shape

        if self.sigmoid:
            sigmoid_pred_T = torch.sigmoid(preds_T.view(-1, H * W * D))

            loss = torch.sum(sigmoid_pred_T *
                             torch.sigmoid(preds_T.view(-1, H * W * D)) -
                             sigmoid_pred_T *
                             torch.sigmoid(preds_S.view(-1, H * W * D)))

            loss = self.loss_weight * loss / (C * N)
        else:
            softmax_pred_T = F.softmax(preds_T.view(-1, H * W * D) / self.tau, dim=1)

            logsoftmax = torch.nn.LogSoftmax(dim=1)
            loss = torch.sum(softmax_pred_T *
                             logsoftmax(preds_T.view(-1, H * W * D) / self.tau) -
                             softmax_pred_T *
                             logsoftmax(preds_S.view(-1, H * W * D) / self.tau)) * (
                                 self.tau**2)

            loss = self.loss_weight * loss / (C * N)

        return loss
