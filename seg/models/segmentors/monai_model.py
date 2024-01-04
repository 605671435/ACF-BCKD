# Copyright (c) OpenMMLab. All rights reserved.
import re
from functools import partial

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torch
from torch import Tensor

from mmengine.registry import MODELS
from mmengine.model import BaseModel, BaseModule
from mmengine.optim import OptimWrapper
from mmengine.model import KaimingInit, ConstantInit
from mmseg.utils import (ForwardResults, ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from mmseg.models.utils import resize
from typing import Dict, Optional, Tuple, Union, Sequence

from monai.inferers import sliding_window_inference
from monai.losses import DeepSupervisionLoss

from ..monai_datapreprocessor import MonaiDataPreProcessor


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class MonaiSegmentor(BaseModule):
    def __init__(self,
                 backbone: ConfigType,
                 roi_shapes: Sequence[int],
                 decoder: OptConfigType = None,
                 ):
        super().__init__()
        self.backbone = MODELS.build(backbone)
        if decoder is not None:
            self.decoder = MODELS.build(decoder)
        else:
            self.decoder = None
        self.roi_shapes = roi_shapes
        self.init_cfg = [
            dict(type=KaimingInit, layer=['_ConvNd', '_ConvTransposeNd']),
            dict(
                type=ConstantInit,
                val=1,
                layer=['_BatchNorm', '_InstanceNorm'])
        ]

    # def init_weights(self):
    #     self.backbone.apply(InitWeights_He(1e-2))
    #     if self.decoder is not None:
    #         self.decoder.apply(InitWeights_He(1e-2))

    def forward(self, x: torch.Tensor, predict: bool = False) -> Sequence[torch.Tensor]:
        x = self.backbone(x)
        if self.decoder is not None:
            x = self.decoder(x)
        if isinstance(x, Sequence) and predict:
            x = x[0]
        #     else:
        #         new_x = []
        #         for x_i in x:
        #             if x_i.shape[-3:] != self.roi_shapes:
        #                 x_i = resize(
        #                     input=x_i,
        #                     size=self.roi_shapes,
        #                     mode='trilinear')
        #                 new_x.append(x_i)
        #         return new_x
        # else:
        #     if x.shape[-3:] != self.roi_shapes:
        #         x = resize(
        #             input=x,
        #             size=self.roi_shapes,
        #             mode='trilinear')
        return x


class MonaiSeg(BaseModel):

    def __init__(self,
                 backbone: ConfigType,
                 loss_functions: ConfigType,
                 num_classes: int,
                 roi_shapes: Sequence[int],
                 vae_loss: bool = False,
                 vae_loss_weight: float = 0.1,
                 data_preprocessor: OptConfigType = dict(type=MonaiDataPreProcessor),
                 decoder: OptConfigType = None,
                 infer_cfg: OptConfigType = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=None)
        self.segmentor = MonaiSegmentor(
            backbone=backbone,
            decoder=decoder,
            roi_shapes=roi_shapes)

        if infer_cfg is not None:
            self.inferer = partial(
                sliding_window_inference,
                roi_size=infer_cfg.inf_size,
                sw_batch_size=infer_cfg.sw_batch_size,
                predictor=self.segmentor,
                overlap=infer_cfg.infer_overlap,
                predict=True)
        else:
            self.inferer = None

        self.do_ds = False
        self.vae_loss = vae_loss
        self.vae_loss_weight = vae_loss_weight
        if isinstance(loss_functions, dict):
            self.do_ds = loss_functions.pop('do_ds', False)
            self.loss_weights = loss_functions.pop('loss_weight', 1.0)
            if self.do_ds is True:
                loss_functions = MODELS.build(loss_functions)
                self.loss_functions = DeepSupervisionLoss(loss=loss_functions)
            else:
                self.loss_functions = MODELS.build(loss_functions)
        elif isinstance(loss_functions, (list, tuple)):
            self.loss_functions = nn.ModuleList()
            self.loss_weights = []
            for loss in loss_functions:
                self.loss_weights.append(loss.pop('loss_weight', 1.0))
                self.loss_functions.append(MODELS.build(loss))
        else:
            raise TypeError(f'loss_functions must be a dict or sequence of dict,\
                but got {type(loss_functions)}')

    def forward(self,
                inputs: Tensor,
                data_samples: dict = None,
                mode: str = 'tensor') -> ForwardResults:
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

        # outputs = self.segmentor(inputs)
        #
        # return outputs

    def loss(self, inputs: Tensor, data_samples: dict) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        logits = self.segmentor(inputs)
        # losses = dict()
        if self.do_ds:
            # calculate loss weight
            num_decoders = len(logits)
            weights = np.array([1 / (2 ** i) for i in range(num_decoders)])
            # weights[-1] = 0
            # weights[0] = 1.0
            weights = weights / weights.sum()
            loss_weights = weights.tolist()
            if self.loss_functions.weights is None:
                self.loss_functions.weights = loss_weights
        #     for i, logit in enumerate(logits):
        #         loss = self.loss_by_feat(logit, data_samples['label'], loss_weight=loss_weights[i])
        #         if i == 0:
        #             losses.update(add_prefix(loss, 'main_head'))
        #         else:
        #             losses.update(add_prefix(loss, f'aux_head{i}'))
        # else:
        #     losses = self.loss_by_feat(logits, data_samples['label'])
        losses = self.loss_by_feat(logits, data_samples['label'])

        return losses

    def loss_by_feat(self, logits: Tensor, label: Tensor) -> dict:
        loss = dict()
        if not isinstance(self.loss_functions, nn.ModuleList):
            loss_functions = [self.loss_functions]
            loss_weights = [self.loss_weights]
        else:
            loss_weights = self.loss_weights
            loss_functions = self.loss_functions
        if self.vae_loss:
            logits, vae_loss = logits
            loss['vae_loss'] = vae_loss * self.vae_loss_weight
        for loss_weight, loss_functions in zip(loss_weights, loss_functions):
            if self.do_ds:
                loss_name = loss_functions.__class__.__name__ + self.loss_functions.loss.__class__.__name__
            else:
                loss_name = loss_functions.__class__.__name__
            loss_name = re.sub('(?<=[a-z])[A-Z]|(?<!^)[A-Z](?=[a-z])', '_\g<0>', loss_name).lower()
            if loss_name not in loss:
                loss[loss_name] = loss_weight * loss_functions(
                    logits, label)
            else:
                loss[loss_name] += loss_weight * loss_functions(
                    logits, label)

        return loss

    def predict(self, inputs: Tensor, data_samples: dict) -> torch.Tensor:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        if self.inferer is not None:
            logits = self.inferer(inputs)
        else:
            logits = self.segmentor(inputs)

        return logits

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation.

        During non-distributed training. If subclasses do not override the
        :meth:`train_step`, :class:`EpochBasedTrainLoop` or
        :class:`IterBasedTrainLoop` will call this method to update model
        parameters. The default parameter update process is as follows:

        1. Calls ``self.data_processor(data, training=False)`` to collect
           batch_inputs and corresponding data_samples(labels).
        2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
           loss
        3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``optim_wrapper.update_params(loss)`` to update model.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses, zero_kwargs=dict(set_to_none=True))
        return log_vars