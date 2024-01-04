# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional
import logging
from collections import defaultdict

import torch
from torch import nn
from mmengine.logging import print_log
from mmengine.model.weight_init import PretrainedInit, initialize, update_init_info
from mmengine.model.wrappers.utils import is_model_wrapper

from razor.registry import MODELS
from mmrazor.models.task_modules.delivery import DistillDeliveryManager
from mmrazor.models.distillers import ConfigurableDistiller as _ConfigurableDistiller
from ..algorithms.base import LossResults

from ..task_modules.recorder import RecorderManager


@MODELS.register_module()
class ConfigurableDistiller(_ConfigurableDistiller):
    def __init__(self,
                 student_recorders: Optional[Dict[str, Dict]] = None,
                 teacher_recorders: Optional[Dict[str, Dict]] = None,
                 distill_deliveries: Optional[Dict[str, Dict]] = None,
                 connectors: Optional[Dict[str, Dict]] = None,
                 distill_losses: Optional[Dict[str, Dict]] = None,
                 loss_forward_mappings: Optional[Dict[str, Dict]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        # The recorder manager is just constructed, but not really initialized
        # yet. Recorder manager initialization needs to input the corresponding
        # model.
        self.student_recorders = RecorderManager(student_recorders)
        self.teacher_recorders = RecorderManager(teacher_recorders)

        self.deliveries = DistillDeliveryManager(distill_deliveries)

        self.distill_losses = self.build_distill_losses(distill_losses)

        self.connectors = self.build_connectors(connectors)

        if loss_forward_mappings:
            # Check if loss_forward_mappings is in the correct format.
            self._check_loss_forward_mappings(self.distill_losses,
                                              loss_forward_mappings,
                                              self.student_recorders,
                                              self.teacher_recorders)
            self.loss_forward_mappings = loss_forward_mappings
        else:
            self.loss_forward_mappings = dict()

    def init_weights(self):
        """Initialize the weights."""

        is_top_level_module = False
        # check if it is top-level module
        if not hasattr(self, '_params_init_info'):
            # The `_params_init_info` is used to record the initialization
            # information of the parameters
            # the key should be the obj:`nn.Parameter` of model and the value
            # should be a dict containing
            # - init_info (str): The string that describes the initialization.
            # - tmp_mean_value (FloatTensor): The mean of the parameter,
            #       which indicates whether the parameter has been modified.
            # this attribute would be deleted after all parameters
            # is initialized.
            self._params_init_info = defaultdict(dict)
            is_top_level_module = True

            # Initialize the `_params_init_info`,
            # When detecting the `tmp_mean_value` of
            # the corresponding parameter is changed, update related
            # initialization information
            for name, param in self.named_parameters():
                self._params_init_info[param][
                    'init_info'] = f'The value is the same before and ' \
                                   f'after calling `init_weights` ' \
                                   f'of {self.__class__.__name__} '
                self._params_init_info[param][
                    'tmp_mean_value'] = param.data.mean().cpu()

            # pass `params_init_info` to all submodules
            # All submodules share the same `params_init_info`,
            # so it will be updated when parameters are
            # modified at any level of the model.
            for sub_module in self.modules():
                sub_module._params_init_info = self._params_init_info

        module_name = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:
                print_log(
                    f'initialize {module_name} with init_cfg {self.init_cfg}',
                    logger='current',
                    level=logging.DEBUG)

                init_cfgs = self.init_cfg
                if isinstance(self.init_cfg, dict):
                    init_cfgs = [self.init_cfg]

                # PretrainedInit has higher priority than any other init_cfg.
                # Therefore we initialize `pretrained_cfg` last to overwrite
                # the previous initialized weights.
                # See details in https://github.com/open-mmlab/mmengine/issues/691 # noqa E501
                other_cfgs = []
                pretrained_cfg = []
                for init_cfg in init_cfgs:
                    assert isinstance(init_cfg, dict)
                    if (init_cfg['type'] == 'Pretrained'
                            or init_cfg['type'] is PretrainedInit):
                        pretrained_cfg.append(init_cfg)
                    else:
                        other_cfgs.append(init_cfg)

                initialize(self, other_cfgs)

            for m in self.children():
                if is_model_wrapper(m) and not hasattr(m, 'init_weights'):
                    m = m.module
                if isinstance(m, nn.ModuleDict):
                    for v in m.values():
                        if hasattr(v, 'init_weights') and not getattr(
                               v, 'is_init', False):
                            v.init_weights()
                            # users may overload the `init_weights`
                            update_init_info(
                                v,
                                init_info=f'Initialized by '
                                          f'user-defined `init_weights`'
                                          f' in {v.__class__.__name__} ')
                if hasattr(m, 'init_weights') and not getattr(
                        m, 'is_init', False):
                    m.init_weights()
                    # users may overload the `init_weights`
                    update_init_info(
                        m,
                        init_info=f'Initialized by '
                        f'user-defined `init_weights`'
                        f' in {m.__class__.__name__} ')
            if self.init_cfg and pretrained_cfg:
                initialize(self, pretrained_cfg)
            self._is_init = True
        else:
            print_log(
                f'init_weights of {self.__class__.__name__} has '
                f'been called more than once.',
                logger='current',
                level=logging.WARNING)

        if is_top_level_module:
            self._dump_init_info()

            for sub_module in self.modules():
                del sub_module._params_init_info

    def compute_distill_losses(self) -> LossResults:
        """Compute distill losses automatically."""
        # Record all computed losses' results.
        losses = dict()
        for loss_name, forward_mappings in self.loss_forward_mappings.items():
            forward_kwargs = dict()
            for forward_key, record in forward_mappings.items():
                forward_var = self.get_record(**record)
                forward_kwargs[forward_key] = forward_var

            loss_module = self.distill_losses[loss_name]
            loss = loss_module(**forward_kwargs)  # type: ignore
            # add computed loss result.
            if isinstance(loss, dict):
                for key, value in loss.items():
                    losses[key] = value
            elif isinstance(loss, (torch.Tensor, tuple)):
                losses[loss_name] = loss
            else:
                raise TypeError(f'type of loss should be dict, tensor or tuple, not {type(loss)}')
        return losses
