# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
from mmengine.model import BaseModel
from mmengine.runner import load_checkpoint
from mmengine.structures import BaseDataElement
from mmengine.config import Config
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.utils import add_prefix
from seg.registry import MODELS as SEG_MODELS
from razor.registry import MODELS
from ...base import BaseAlgorithm, LossResults
from .single_teacher_distill import SingleTeacherDistill


@MODELS.register_module()
class SingleTeacherDistill(SingleTeacherDistill):

    def loss(
        self,
        batch_inputs: torch.Tensor,
        data_samples: Optional[List[BaseDataElement]] = None,
    ) -> LossResults:
        """Calculate losses from a batch of inputs and data samples."""

        losses = dict()

        # If the `override_data` of a delivery is False, the delivery will
        # record the origin data.
        self.distiller.set_deliveries_override(False)
        if self.teacher_trainable:
            with self.distiller.teacher_recorders, self.distiller.deliveries:
                teacher_losses = self.teacher(
                    batch_inputs, data_samples, mode='loss')

            losses.update(add_prefix(teacher_losses, 'teacher'))
        else:
            with self.distiller.teacher_recorders, self.distiller.deliveries:
                with torch.no_grad():
                    _ = self.teacher(batch_inputs, data_samples, mode='loss')

        # If the `override_data` of a delivery is True, the delivery will
        # override the origin data with the recorded data.
        self.distiller.set_deliveries_override(True)
        # Original task loss will not be used during some pretraining process.
        if self.calculate_student_loss:
            with self.distiller.student_recorders, self.distiller.deliveries:
                student_losses = self.student(
                    batch_inputs, data_samples, mode='loss')
            losses.update(add_prefix(student_losses, 'student'))
        else:
            with self.distiller.student_recorders, self.distiller.deliveries:
                if self.student_trainable:
                    _ = self.student(batch_inputs, data_samples, mode='loss')
                else:
                    with torch.no_grad():
                        _ = self.student(
                            batch_inputs, data_samples, mode='loss')

        if not self.distillation_stopped:
            # Automatically compute distill losses based on
            # `loss_forward_mappings`.
            # The required data already exists in the recorders.
            distill_losses = self.distiller.compute_distill_losses()
            losses.update(add_prefix(distill_losses, 'distill'))

        return losses

