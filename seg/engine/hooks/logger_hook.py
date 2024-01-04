# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import os.path as osp
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch

from mmengine.fileio import FileClient, dump
from mmengine.fileio.io import get_file_backend
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.registry import HOOKS
from mmengine.utils import is_seq_of, scandir
from mmengine.hooks import LoggerHook
DATA_BATCH = Optional[Union[dict, tuple, list]]
SUFFIX_TYPE = Union[Sequence[str], str]


@HOOKS.register_module()
class MyLoggerHook(LoggerHook):
    def __init__(self, val_interval, **kwargs):
        super(MyLoggerHook, self).__init__(**kwargs)
        self.val_interval = val_interval

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """Record logs after training iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict tuple or list, optional): Data from dataloader.
            outputs (dict, optional): Outputs from model.
        """
        # Print experiment name every n iterations.
        if self.every_n_train_iters(
                runner, self.interval_exp_name):
            exp_info = f'Exp name: {runner.experiment_name}'
            runner.logger.info(exp_info)
        if self.every_n_inner_iters(batch_idx, self.interval):
            tag, log_str = runner.log_processor.get_log_after_iter(
                runner, batch_idx, 'train')
        elif (self.end_of_epoch(runner.train_dataloader, batch_idx)
              and (not self.ignore_last
                   or len(runner.train_dataloader) <= self.interval)):
            # `runner.max_iters` may not be divisible by `self.interval`. if
            # `self.ignore_last==True`, the log of remaining iterations will
            # be recorded (Epoch [4][1000/1007], the logs of 998-1007
            # iterations will be recorded).
            tag, log_str = runner.log_processor.get_log_after_iter(
                runner, batch_idx, 'train')
        else:
            return
        runner.logger.info(log_str)
        runner.visualizer.add_scalars(
            tag, step=runner.iter + 1, file_path=self.json_log_path)

    def after_val_iter(self,
                       runner,
                       batch_idx: int,
                       data_batch: DATA_BATCH = None,
                       outputs: Optional[Sequence] = None) -> None:
        """Record logs after validation iteration.

        Args:
            runner (Runner): The runner of the validation process.
            batch_idx (int): The index of the current batch in the validation
                loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
                Defaults to None.
            outputs (sequence, optional): Outputs from model.
        """
        if self.every_n_inner_iters(batch_idx, self.val_interval):
            _, log_str = runner.log_processor.get_log_after_iter(
                runner, batch_idx, 'val')
            runner.logger.info(log_str)

    def after_test_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None,
                        outputs: Optional[Sequence] = None) -> None:
        """Record logs after testing iteration.

        Args:
            runner (Runner): The runner of the testing process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (sequence, optional): Outputs from model.
        """
        if self.every_n_inner_iters(batch_idx, self.val_interval):
            _, log_str = runner.log_processor.get_log_after_iter(
                runner, batch_idx, 'test')
            runner.logger.info(log_str)

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each validation epoch.

        Args:
            runner (Runner): The runner of the validation process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on validation dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        tag, log_str = runner.log_processor.get_log_after_epoch(
            runner, len(runner.val_dataloader), 'val')
        runner.logger.info(log_str)
        # remove class-wise scalars
        new_tag = OrderedDict({'nothing': 1})
        for k, v in tag.items():
            if k.find('(') == -1:
                # if k == 'Dice':
                #     new_tag['DICE'] = v
                new_tag[k] = v
        if self.log_metric_by_epoch:
            # Accessing the epoch attribute of the runner will trigger
            # the construction of the train_loop. Therefore, to avoid
            # triggering the construction of the train_loop during
            # validation, check before accessing the epoch.
            if (isinstance(runner._train_loop, dict)
                    or runner._train_loop is None):
                epoch = 0
            else:
                epoch = runner.epoch
            runner.visualizer.add_scalars(
                new_tag, step=epoch, file_path=self.json_log_path)
        else:
            if (isinstance(runner._train_loop, dict)
                    or runner._train_loop is None):
                iter = 0
            else:
                iter = runner.iter
            runner.visualizer.add_scalars(
                new_tag, step=iter, file_path=self.json_log_path)