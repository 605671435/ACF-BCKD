# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader

from mmengine.evaluator import Evaluator
from seg.registry import LOOPS, EVALUATOR
from mmengine.runner.amp import autocast
from mmengine.runner import BaseLoop
from mmengine.registry import DATASETS
from mmengine.logging import MMLogger, print_log

class ValLoop(BaseLoop):
    """Loop for validation.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False) -> None:
        self._runner = runner
        dataloader_cfg = copy.deepcopy(dataloader)
        case_datasets = []
        # build dataset
        dataset_cfg = dataloader_cfg.pop('dataset')
        if isinstance(dataset_cfg, dict):
            dataset = DATASETS.build(dataset_cfg)
            if hasattr(dataset, 'full_init'):
                dataset.full_init()

        if not hasattr(dataset, 'get_subset'):
            dataset = dataset.dataset
        case_nums = dataset.metainfo['case_nums']
        for i, (case, nums) in enumerate(case_nums.items()):
            if len(dataset) - nums > 0:
                remain_dataset = dataset.get_subset(nums - len(dataset))
            else:
                case_datasets.append(dataset)
                break
            sub_dataset = dataset.get_subset(nums)
            case_datasets.append(sub_dataset)
            dataset = remain_dataset

        self.case_dataloaders = []
        assert isinstance(dataloader, dict)
        # Determine whether or not different ranks use different seed.
        diff_rank_seed = runner._randomness_cfg.get(
            'diff_rank_seed', False)
        for case_dataset in case_datasets:
            _dataloader = copy.deepcopy(dataloader)
            _dataloader.update(dataset=case_dataset)
            self.case_dataloaders.append(runner.build_dataloader(
                _dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed))

        if isinstance(evaluator, dict) or isinstance(evaluator, list):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
        else:
            assert isinstance(evaluator, Evaluator), (
                'evaluator must be one of dict, list or Evaluator instance, '
                f'but got {type(evaluator)}.')
            self.evaluator = evaluator  # type: ignore
        if hasattr(self.case_dataloaders[0].dataset, 'metainfo'):
            self.evaluator.dataset_meta = self.case_dataloaders[0].dataset.metainfo
            self.runner.visualizer.dataset_meta = \
                self.case_dataloaders[0].dataset.metainfo
        else:
            print_log(
                f'Dataset {self.case_dataloaders[0].dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in evaluator, metric and '
                'visualizer will be None.',
                logger='current',
                level=logging.WARNING)
        self.fp16 = fp16

    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()

        case_metrics = []
        logger: MMLogger = MMLogger.get_current_instance()
        for i, case_dataloader in enumerate(self.case_dataloaders):
            logger.info(
                f'----------- Testing on case: [{i + 1}/{len(self.case_dataloaders)}] ----------- ')
            self.dataloader = case_dataloader
            assert isinstance(self.dataloader, DataLoader)
            for idx, data_batch in enumerate(self.dataloader):
                self.run_iter(idx, data_batch)
            # compute metrics
            case_metrics.append(self.evaluator.evaluate(len(self.dataloader.dataset)))
        metrics = dict()
        for key in case_metrics[0].keys():
            metrics[key] = np.round(np.nanmean([case_metric[key] for case_metric in case_metrics]), 2)
        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')
        return metrics

    def run_case(self):
        """Launch validation."""
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.val_step(data_batch)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)


class TestLoop(BaseLoop):
    """Loop for test.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 testing. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False):
        super().__init__(runner, dataloader)

        self._runner = runner
        dataloader_cfg = copy.deepcopy(dataloader)
        case_datasets = []
        # build dataset
        dataset_cfg = dataloader_cfg.pop('dataset')
        if isinstance(dataset_cfg, dict):
            dataset = DATASETS.build(dataset_cfg)
            if hasattr(dataset, 'full_init'):
                dataset.full_init()

        if not hasattr(dataset, 'get_subset'):
            dataset = dataset.dataset
        case_nums = dataset.metainfo['case_nums']
        for i, (case, nums) in enumerate(case_nums.items()):
            if len(dataset) - nums > 0:
                remain_dataset = dataset.get_subset(nums - len(dataset))
            else:
                case_datasets.append(dataset)
                break
            sub_dataset = dataset.get_subset(nums)
            case_datasets.append(sub_dataset)
            dataset = remain_dataset

        self.case_dataloaders = []
        assert isinstance(dataloader, dict)
        # Determine whether or not different ranks use different seed.
        diff_rank_seed = runner._randomness_cfg.get(
            'diff_rank_seed', False)
        for case_dataset in case_datasets:
            _dataloader = copy.deepcopy(dataloader)
            _dataloader.update(dataset=case_dataset)
            self.case_dataloaders.append(runner.build_dataloader(
                _dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed))

        if isinstance(evaluator, dict) or isinstance(evaluator, list):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
        else:
            self.evaluator = evaluator  # type: ignore
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in evaluator, metric and '
                'visualizer will be None.',
                logger='current',
                level=logging.WARNING)
        self.fp16 = fp16

    def run(self) -> dict:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()

        case_metrics = []
        logger: MMLogger = MMLogger.get_current_instance()
        for i, case_dataloader in enumerate(self.case_dataloaders):
            logger.info(
                f'----------- Testing on case: [{i + 1}/{len(self.case_dataloaders)}] ----------- ')
            self.dataloader = case_dataloader
            for idx, data_batch in enumerate(self.dataloader):
                self.run_iter(idx, data_batch)
            # compute metrics
            case_metrics.append(self.evaluator.evaluate(len(self.dataloader.dataset)))
        metrics = dict()
        for key in case_metrics[0].keys():
            metrics[key] = np.round(np.nanmean([case_metric[key] for case_metric in case_metrics]), 2)
        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        # predictions should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.test_step(data_batch)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
# class CaseTestLoop(TestLoop):
#
#     def run(self) -> dict:
#         """Launch test."""
#         self.runner.call_hook('before_test')
#         self.runner.call_hook('before_test_epoch')
#         self.runner.model.eval()
#         # outputs = []
#         for idx, data_batch in enumerate(self.dataloader):
#             if idx == 0:
#                 outputs = self.run_iter(idx, data_batch)
#             else:
#                 outputs.extend(self.run_iter(idx, data_batch))
#
#         # compute metrics
#         metrics = self.evaluator.offline_evaluate(data_samples=outputs)
#         self.runner.call_hook('after_test_epoch', metrics=metrics)
#         self.runner.call_hook('after_test')
#         return metrics
#
#     @torch.no_grad()
#     def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
#         """Iterate one mini-batch.
#
#         Args:
#             data_batch (Sequence[dict]): Batch of data from dataloader.
#         """
#         self.runner.call_hook(
#             'before_test_iter', batch_idx=idx, data_batch=data_batch)
#         # predictions should be sequence of BaseDataElement
#         with autocast(enabled=self.fp16):
#             outputs = self.runner.model.test_step(data_batch)
#         # self.evaluator.process(data_samples=outputs, data_batch=data_batch)
#         self.runner.call_hook(
#             'after_test_iter',
#             batch_idx=idx,
#             data_batch=data_batch,
#             outputs=outputs)
#         return outputs