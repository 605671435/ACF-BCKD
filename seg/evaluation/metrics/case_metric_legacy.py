# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import os.path as osp
from functools import partial
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist, ProgressBar, track_parallel_progress
from PIL import Image
from prettytable import PrettyTable
from .confusion_matrix import *

def calculate_metric_percase(cls_index, pred, gt, metrics):
    confusion_matrix = ConfusionMatrix(test=pred == cls_index, reference=gt == cls_index)
    rets = []
    for metric in metrics:
        ret = ALL_METRICS[metric](pred, gt, confusion_matrix)
        rets.append(ret)
    return rets

class CaseMetric(BaseMetric):
    """IoU evaluation metric.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mIoU', 'mDice' and 'mFscore'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 ignore_index: int = 255,
                 case_metrics: List[str] = ['Dice', 'Jaccard', 'HD95', 'ASD'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        for metric in case_metrics:
            assert metric in list(ALL_METRICS.keys())
        self.metrics = case_metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].cpu()
            label = data_sample['gt_sem_seg']['data'].to(
                pred_label)
            # self.results.append(
            #     dict({
            #         'pred': pred_label.numpy(),
            #         'label': label.numpy()}))
            if len(self.results) == 0:
                self.results.append(
                    dict({
                        'pred': pred_label.numpy(),
                        'label': label.numpy()}))
            else:
                self.results[0]['pred'] = np.concatenate([self.results[0]['pred'], pred_label.numpy()], 0)
                self.results[0]['label'] = np.concatenate([self.results[0]['label'], label.numpy()], 0)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        class_names = self.dataset_meta['classes']
        num_classes = len(class_names)

        total_pred_label, total_label = results[0]['pred'], results[0]['label']
        ret_metrics = self.label_to_metrics(total_pred_label,
                                            total_label,
                                            num_classes,
                                            self.metrics)
        # summary table
        # ret_metrics_summary = OrderedDict({
        #     'Dice': np.round(np.nanmean(ret_metrics['Dice']) * 100, 2),
        #     'Jaccard': np.round(np.nanmean(ret_metrics['Jaccard']) * 100, 2),
        #     'HD95': np.round(np.nanmean(ret_metrics['HD95']), 2)
        # })

        ret_metrics_summary = OrderedDict({
            metric: np.round(np.nanmean(ret_metrics[metric]), 2)
            for metric in self.metrics})

        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                metrics[key] = val
            else:
                metrics['m' + key] = val

        # each class table
        ret_metrics_class = OrderedDict({
            metric: np.round(ret_metrics[metric], 2)
            for metric in self.metrics})
        # ret_metrics_class = OrderedDict({
        #     'Dice': np.round(ret_metrics['Dice'] * 100, 2),
        #     'Jaccard': np.round(ret_metrics['Jaccard'] * 100, 2),
        #     'HD95': np.round(ret_metrics['HD95'], 2)
        # })

        for metric in self.metrics:  # ['Dice', 'Jaccard', 'HD95']
            for class_key, metric_value in zip(class_names[1:], ret_metrics_class[metric]):
                metrics[f'{metric} ({class_key})'] = np.round(metric_value, 4)

        # for metric in self.metrics:
        #     metric = metric.split('m')[-1] if metric.startswith('m') else metric
        #     for class_key, metric_value in zip(class_names[1:], ret_metrics_class[metric]):
        #         metrics[f'{metric} ({class_key})'] = np.round(metric_value, 4)

        ret_metrics_class.update({'Class': class_names[1:]})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics

    @staticmethod
    def label_to_metrics(prediction: np.ndarray, target: np.ndarray,
                         num_classes: int, metrics: List[str]):
        try:
            from medpy import metric
        except ImportError:
            warnings.warn("medpy not installed, cannot compute hd_distance")
            return list()

        # def calculate_metric_percase(cls_index, pred, gt, metrics):
        #     confusion_matrix = ConfusionMatrix(test=pred == cls_index, reference=gt == cls_index)
        #     rets = []
        #     for metric in metrics:
        #         # dc = dice(pred, gt, confusion_matrix)
        #         # jc = jaccard(pred, gt, confusion_matrix)
        #         # hd95 = hausdorff_distance_95(pred, gt, confusion_matrix)
        #         ret = ALL_METRICS[metric](pred, gt, confusion_matrix)
        #         rets.append(ret)
        #     # return dc, jc, hd95
        #     return rets
        # dice_list = []
        # jaccard_list = []
        # hd95_list = []
        # pb = ProgressBar(num_classes - 1)

        ret = track_parallel_progress(
            partial(calculate_metric_percase, pred=prediction, gt=target, metrics=metrics),
            [i for i in range(1, num_classes)],
            nproc=num_classes - 1)

        results = list(zip(*ret))

        # for c in range(1, num_classes):
        #     dc, jc, hd95 = calculate_metric_percase(prediction == c, target == c)
        #     dice_list.append(dc)
        #     jaccard_list.append(jc)
        #     hd95_list.append(hd95)
        #     pb.update()
        # ret_metrics = OrderedDict({
        #     'Dice': np.array(dice_list),
        #     'Jaccard': np.array(jaccard_list),
        #     'HD95': np.array(hd95_list)})
        ret_metrics = OrderedDict({
            metric: np.array(results[i])
            for i, metric in enumerate(metrics)})
        return ret_metrics

