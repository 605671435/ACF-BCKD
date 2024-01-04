# -*- coding: utf-8 -*-
# copyright from:
# https://github.com/HiLab-git/LCOVNet-and-KD/blob/ee8ab5e0060d6270abf881a0fcca98f970cfff80/pymic/loss/seg/kd.py
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch import Tensor
from torch import nn, Tensor
from monai.networks import one_hot

class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # if len(target.shape) == len(input.shape):
        #     assert target.shape[1] == 1
        #     target = target[:, 0]
        return super().forward(input, target)


class MSKDCAKDLoss(nn.Module):
    def __init__(self, loss_weight, sigmoid=False, cakd_weight=1.0, fnkd_weight=1.0):
        super(MSKDCAKDLoss, self).__init__()
        ce_kwargs = {}
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.loss_weight = loss_weight
        self.cakd_weight = cakd_weight
        self.fnkd_weight = fnkd_weight
        self.sigmoid = sigmoid

    def forward(self, student_outputs, teacher_outputs, student_features, teacher_features):
        # loss = torch.tensor(0.).cuda()
        # w = [0.4, 0.2, 0.2, 0.2]
        # for i in range(0, 4):
        #     loss += w[i] * (0.1 * self.CAKD(student_outputs[i], teacher_outputs[i])
        #                     + 0.2 * self.FNKD(student_outputs[i], teacher_outputs[i], student_outputs[i + 4],
        #                                       teacher_outputs[i + 4]))

        cakd_loss = self.cakd_weight * self.CAKD(student_outputs, teacher_outputs)

        fnkd_loss = self.fnkd_weight * self.FNKD(
            student_outputs, teacher_outputs, student_features, teacher_features)

        loss = cakd_loss + fnkd_loss
        return loss * self.loss_weight

    def CAKD(self, student_outputs, teacher_outputs):
        [B, C, D, W, H] = student_outputs.shape

        if self.sigmoid:
            student_outputs = torch.sigmoid(student_outputs)
            teacher_outputs = torch.sigmoid(teacher_outputs)
        else:
            student_outputs = F.softmax(student_outputs, dim=1)
            teacher_outputs = F.softmax(teacher_outputs, dim=1)

        student_outputs = student_outputs.reshape(B, C, D * W * H)
        teacher_outputs = teacher_outputs.reshape(B, C, D * W * H)

        with autocast(enabled=False):
            student_outputs = torch.bmm(student_outputs, student_outputs.permute(
                0, 2, 1))
            teacher_outputs = torch.bmm(teacher_outputs, teacher_outputs.permute(
                0, 2, 1))
        Similarity_loss = F.cosine_similarity(student_outputs[0, :, :], teacher_outputs[0, :, :], dim=0)
        for b in range(1, B):
            Similarity_loss += F.cosine_similarity(student_outputs[b, :, :], teacher_outputs[b, :, :], dim=0)
        Similarity_loss = Similarity_loss / B
        # Similarity_loss = (F.cosine_similarity(student_outputs[0, :, :], teacher_outputs[0, :, :], dim=0) +
        #                    F.cosine_similarity(
        #                        student_outputs[1, :, :], teacher_outputs[1, :, :], dim=0)) / 2
        loss = -torch.mean(Similarity_loss)  # loss = 0 fully same
        return loss

    def FNKD(self, student_outputs, teacher_outputs, student_feature, teacher_feature):
        num_classes = student_outputs.shape[1]
        student_L2norm = torch.norm(student_feature)
        teacher_L2norm = torch.norm(teacher_feature)
        if self.sigmoid:
            q_fn = F.sigmoid(teacher_outputs / teacher_L2norm)
            to_kd = F.sigmoid(student_outputs / student_L2norm)
        else:
            q_fn = F.softmax(teacher_outputs / teacher_L2norm, dim=1)
            to_kd = F.log_softmax(student_outputs / student_L2norm, dim=1)

        KD_ce_loss = self.ce(
            to_kd, q_fn)
        # KD_ce_loss = self.ce(
        #     q_fn, to_kd[:, 0].long())
        return KD_ce_loss
