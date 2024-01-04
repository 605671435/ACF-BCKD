# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import copy
from inspect import signature
from typing import Dict, List, Optional, Union

from mmengine.model import BaseModel
from torch import nn

from razor.registry import MODELS
from ..algorithms.base import LossResults
from razor.models.distillers.configurable_distiller import ConfigurableDistiller
from mmrazor.models.task_modules import DistillDeliveryManager
from razor.models.task_modules import RecorderManager

@MODELS.register_module()
class EXDistiller(ConfigurableDistiller):
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
    def get_record(self,
                   recorder: str,
                   from_student: bool,
                   record_idx: int or List = 0,
                   data_idx: Optional[int] = None,
                   connector: Optional[str] = None,
                   connector_idx: Optional[int] = None) -> List:
        """According to each item in ``record_infos``, get the corresponding
        record in ``recorder_manager``."""

        if from_student:
            recorder_ = self.student_recorders.get_recorder(recorder)
        else:
            recorder_ = self.teacher_recorders.get_recorder(recorder)
        if isinstance(record_idx, List):
            record_data = recorder_.get_record_data(record_idx[0], data_idx)
            for idx in record_idx[1:]:
                record_data += recorder_.get_record_data(idx, data_idx)
        else:
            record_data = recorder_.get_record_data(record_idx, data_idx)

        if connector:
            record_data = self.connectors[connector](record_data)
        if connector_idx is not None:
            record_data = record_data[connector_idx]

        return record_data

    def compute_distill_losses(self) -> LossResults:
        """Compute distill losses automatically."""
        # Record all computed losses' results.
        losses = dict()
        for loss_name, forward_mappings in self.loss_forward_mappings.items():
            forward_kwargs = dict()
            for forward_key, record in forward_mappings.items():
                if record['recorder'] is not None and \
                        isinstance(record['recorder'], List):
                    _forward_var = []
                    for _recorder in record.recorder:
                        _record = copy.deepcopy(record)
                        _record['recorder'] = _recorder
                        _forward_var.append(self.get_record(**_record))
                    forward_var = _forward_var[0]
                    for var in range(1, len(_forward_var)):
                        forward_var = forward_var + _forward_var[var]
                else:
                    forward_var = self.get_record(**record)
                forward_kwargs[forward_key] = forward_var

            loss_module = self.distill_losses[loss_name]
            loss = loss_module(**forward_kwargs)  # type: ignore
            # add computed loss result.
            losses[loss_name] = loss

        return losses

    def _check_loss_forward_mappings(
            self, losses: nn.ModuleDict, loss_forward_mappings: Dict[str,
                                                                     Dict],
            student_recorders: RecorderManager,
            teacher_recorders: RecorderManager) -> None:
        """Check if ``loss_forward_mappings`` is in the correct format."""

        if not isinstance(loss_forward_mappings, dict):
            raise TypeError(
                'loss_forward_mappings should be a dict instance, but got'
                f'{type(loss_forward_mappings)}')

        for loss_name, forward_mappings in loss_forward_mappings.items():
            assert loss_name in losses, \
                f'"{loss_name}" is not in distill losses. The keys of ' \
                'loss_forward_kwargs must match the keys of distill_losses.'

            if not isinstance(forward_mappings, dict):
                raise TypeError(
                    'Each item of loss_forward_mappings should be a dict '
                    f'instance, but got {type(forward_mappings)}')

            loss_module = losses[loss_name]
            loss_forward_params = signature(loss_module.forward).parameters
            loss_forward_keys = loss_forward_params.keys()
            # Allow default params.
            # Check non-default params, not len(params).

            for forward_key, record_info in forward_mappings.items():
                assert forward_key in loss_forward_keys, \
                    f'{forward_key} is not in the signature of \
                    {type(loss_module).__name__} forward, \
                    please check your config.'

                if (loss_forward_params[forward_key].default !=
                        loss_forward_params[forward_key].empty):
                    # default params without check
                    continue

                assert 'recorder' in record_info, \
                    'Each item of loss_forward_mappings should have ' \
                    '"recorder", pls check your config.'

                assert 'from_student' in record_info, \
                    'Each item of loss_forward_mappings should have ' \
                    '"from_student", pls check your config.'

                recorder: List = record_info['recorder']
                if not isinstance(recorder, List):
                    recorder = [recorder, ]
                from_student: bool = record_info['from_student']

                if not isinstance(from_student, bool):
                    raise TypeError(f'from_student should be a bool instance, '
                                    f'but got {type(from_student)}')

                if from_student:
                    for _recorder in recorder:
                        assert _recorder in student_recorders.recorders, \
                            f'For {forward_key}, "{_recorder}" must be in \
                            `student_recorders`.'

                else:
                    for _recorder in recorder:
                        assert _recorder in teacher_recorders.recorders, \
                            f'For {forward_key}, "{_recorder}" must be in \
                            `teacher_recorders`.'

                if 'connector' in record_info:
                    connector: str = record_info['connector']
                    assert connector in self.connectors, \
                        f'{connector} must be in "connectors".'