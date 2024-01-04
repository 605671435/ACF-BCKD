# Copyright (c) OpenMMLab. All rights reserved.
import typing
import os.path as osp
import warnings
import datetime
import logging
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Sequence, Any

from torch import Tensor
import numpy
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from seg.registry import METRICS
import numpy as np
from prettytable import PrettyTable
import torch

from mmengine.visualization import Visualizer
from mmengine.utils import Timer

from mmengine.dist import (broadcast_object_list, collect_results,
                           is_main_process)
from mmengine.structures import BaseDataElement

@METRICS.register_module()
class PerCaseMetric(BaseMetric):

    def __init__(self,
                 ignore_index: int = 255,
                 iou_metrics: List[str] = ['mIoU'],
                 hd_metric: bool = False,
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 spacing_path: Optional[str] = None,
                 split_for_case: Optional[str] = '_',
                 print_per_case: Optional[bool] = False,
                 return_per_case: Optional[bool] = False,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.metrics = iou_metrics
        self.hd_metric = hd_metric
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        self.format_only = format_only
        self.split_for_case = split_for_case
        self.print_per_case = print_per_case
        self.return_per_case = return_per_case
        self.pre_results = dict()
        self.processing_case = None

        if self.hd_metric:
            self.spacing_path = spacing_path

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        class_names = self.dataset_meta['classes']
        case_nums = self.dataset_meta['case_nums']
        if self.processing_case is None:
            self.processing_case = dict(case_i=0, len=len([k for k in case_nums.values()][0]), now=1)
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data']
            case_index = osp.split(data_sample['img_path'])[-1].split(self.split_for_case)[0]
            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample['gt_sem_seg']['data'].to(
                    pred_label)
                if len(self.pre_results.keys()) == 0:
                    self.pre_results = dict(pred=pred_label, gt=label, case_index=case_index)
                else:
                    self.pre_results.update(pred=torch.cat([self.pre_results['pred'], pred_label]))
                    self.pre_results.update(gt=torch.cat([self.pre_results['gt'], pred_label]))
                    # self.results[case_index] = dict(pred=torch.cat([self.results[case_index]['pred'], pred_label]),
                    #                                 gt=torch.cat([self.results[case_index]['gt'], label]))
                now = self.processing_case['now']
                if now == self.processing_case['len']:
                    case_i = self.processing_case['case_i']
                    self.results.append(self.compute_per_case(pred=self.pre_results['pred'],
                                                              gt=self.pre_results['gt'],
                                                              class_names=class_names))
                    self.pre_results.clear()
                    if case_i < len(case_nums):
                        self.processing_case.update(len=[k for k in case_nums.values()][case_i], now=1)
                else:
                    self.processing_case.update(now=self.processing_case['now'] + 1)
                # self.results.append(dict(pred=pred_label, gt=label, case_index=case_index))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()

        visualizer = Visualizer.get_current_instance()
        uniq_case = np.unique([ret['case_index'] for ret in results])
        num_cases = uniq_case.shape[0]
        class_names = self.dataset_meta['classes']
        num_classes = len(class_names)
        case_results = [[ret for ret in results if ret['case_index'] == case_index] for case_index in uniq_case]

        cases_gt = []
        cases_pred = []
        for i, case in enumerate(case_results):
            cases_gt.append(torch.stack([case_slice['gt'] for case_slice in case]))
            cases_pred.append(torch.stack([case_slice['pred'] for case_slice in case]))

        list_metrics, hd_metrics = self.compute_per_case(cases_pred, cases_gt, class_names, num_cases, uniq_case)

        # summary table
        ret_metrics = dict()
        metrics = dict()
        per_class_dict = dict()
        for key, val in list_metrics[0].items():
            if key == 'aAcc':
                continue
            else:
                per_class = np.array([case[key] for case in list_metrics])
                ret_metrics[key] = [np.round(np.nanmean(per_class[:, c]) * 100, 2)
                                    for c in range(num_classes)]
                metrics['m' + key] = np.round(np.nanmean(per_class) * 100, 2)
                per_class_dict[key] = per_class

        if self.hd_metric:
            ret_metrics['HD'] = np.round(np.nanmean(hd_metrics, axis=0), 2)
            metrics['mHD'] = np.round(np.mean(ret_metrics['HD']), 2)

        # each class table
        ret_metrics_class = OrderedDict({
            ret_metric: ret_metric_value
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        if self.return_per_case:
            for metric in self.metrics:
                metric = metric.split('m')[-1]
                for i, (class_key, metric_value) in enumerate(zip(class_names,
                                                                  ret_metrics_class[metric])):
                    metrics[f'{metric} ({class_key})'] = metric_value

        vis_metric = self.metrics[0].split('m')[-1]
        vis_metrics_class = OrderedDict({
            f'{vis_metric} ({class_key})': metric_value
            for i, (class_key, metric_value) in enumerate(zip(class_names,
                                                              ret_metrics_class[vis_metric]))
        })

        ret_metrics_class.update({'Class': [class_name for i, class_name in enumerate(class_names)]})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            if key != 'Acc':
                class_table_data.add_column(key, val)

        # each case table
        if self.print_per_case:
            metric_table_data = PrettyTable()
            metric_table_data.add_column(f'Class \ {vis_metric}(case)',
                                         ret_metrics_class['Class'] + ['average', ])
            for i in range(num_cases):
                case_metric = np.append(per_class_dict[vis_metric][i],
                                        np.nanmean(per_class_dict[vis_metric][i]))
                metric_table_data.add_column(f'{vis_metric}({i + 1})',
                                             np.round(case_metric * 100, 2))
            metric_table_data.add_column(f'm{vis_metric}',
                                         ret_metrics_class[vis_metric] + ['-', ])
            print_log('\n' + metric_table_data.get_string(), logger=logger)

            if self.hd_metric:
                hd_table_data = PrettyTable()
                hd_table_data.add_column('Class \ HD(case)',
                                         ret_metrics_class['Class'] + ['average', ])
                for i in range(num_cases):
                    case_hd = np.append(hd_metrics[i], np.nanmean(hd_metrics[i]))
                    hd_table_data.add_column(f'HD({i + 1})', np.round(case_hd, 2))
                hd_table_data.add_column('mHD', ret_metrics_class['HD'].tolist() + ['-', ])
                print_log('\n' + hd_table_data.get_string(), logger=logger)

        print_log('per class results average on case:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        visualizer.add_scalars(vis_metrics_class)

        return metrics

    def compute_per_case(self, pred, gt, class_names):
        num_classes = len(class_names)
        logger: MMLogger = MMLogger.get_current_instance()
        if self.hd_metric:
            try:
                from medpy.metric.binary import hd95
            except ImportError:
                warnings.warn("medpy not installed, cannot compute hd_distance")
            hd_metrics = []
            list_spacing = np.loadtxt(self.spacing_path, dtype=str)
            dict_spacing = dict({f'{list_spacing[i, 0]}': [float(fg) for fg in list_spacing[i, 1:]]
                                 for i in range(list_spacing.shape[0])})
        else:
            hd_metrics = None
            dict_spacing = None
        list_metrics = []

        intersect_and_union = self.intersect_and_union(pred,
                                                       gt,
                                                       num_classes=len(class_names),
                                                       ignore_index=self.ignore_index)
        list_metrics.append(self.total_area_to_metrics(
            intersect_and_union[0], intersect_and_union[1], intersect_and_union[2],
            intersect_and_union[3], self.metrics, self.nan_to_num, self.beta))
        pred = pred.numpy()
        gt = gt.numpy()
        if self.hd_metric:
            hd_time = Timer()
            per_hd_metrics = []
            for c in range(num_classes):
                pred = np.zeros_like(pred)
                gt = np.zeros_like(gt)
                pred[pred == c] = 1
                gt[gt == c] = 1
                if pred.sum() > 0 and gt.sum() > 0 and c != self.ignore_index:
                    cur_hd = hd95(pred, gt)
                    # cur_hd = np.random.randint(low=0, high=100, size=1)[0]
                # elif pred.sum() == 0 and gt.sum() == 0:
                else:
                    cur_hd = np.nan
                # else:
                #     inf = np.sum(np.array(cur_pre.shape) * np.array(dict_spacing['img' + uniq_case[i]]))
                #     cur_hd = inf
                per_hd_metrics.append(cur_hd)

                hd_elapsed = hd_time.since_start()
                hd_percent = (c + 1) / float(num_classes)
                hd_eta = int(hd_elapsed * (1 - hd_percent) / hd_percent + 0.5)
                hd_elapsed = str(datetime.timedelta(seconds=int(hd_elapsed)))
                hd_eta = str(datetime.timedelta(seconds=int(hd_eta)))
                print_log(f'Compute HD per class [{c + 1}/{num_classes}] '
                          f'elapsed: {hd_elapsed} eta: {hd_eta} '
                          f'class name: {class_names[c]} '
                          f'HD: {cur_hd:.2f}',
                          logger)
            hd_metrics.append(per_hd_metrics)

        return dict(list_metrics=list_metrics, hd_metrics=hd_metrics)

    # def compute_per_case(self, cases_pred, cases_gt, class_names, num_cases, uniq_case):
    #     num_classes = len(class_names)
    #     logger: MMLogger = MMLogger.get_current_instance()
    #     if self.hd_metric:
    #         try:
    #             from medpy.metric.binary import hd95
    #         except ImportError:
    #             warnings.warn("medpy not installed, cannot compute hd_distance")
    #         hd_metrics = []
    #         list_spacing = np.loadtxt(self.spacing_path, dtype=str)
    #         dict_spacing = dict({f'{list_spacing[i, 0]}': [float(fg) for fg in list_spacing[i, 1:]]
    #                              for i in range(list_spacing.shape[0])})
    #     else:
    #         hd_metrics = None
    #         dict_spacing = None
    #     list_metrics = []
    #     case_time = Timer()
    #     for i, (cur_pre, cur_tar) in enumerate(zip(cases_pred, cases_gt)):
    #         print_log(f'Processing with case{i + 1}...', logger)
    #         intersect_and_union = self.intersect_and_union(cur_pre,
    #                                                        cur_tar,
    #                                                        num_classes=len(class_names),
    #                                                        ignore_index=self.ignore_index)
    #         list_metrics.append(self.total_area_to_metrics(
    #             intersect_and_union[0], intersect_and_union[1], intersect_and_union[2],
    #             intersect_and_union[3], self.metrics, self.nan_to_num, self.beta))
    #         cur_pre = cur_pre.numpy()
    #         cur_tar = cur_tar.numpy()
    #         if self.hd_metric:
    #             hd_time = Timer()
    #             per_hd_metrics = []
    #             for c in range(num_classes):
    #                 pred = np.zeros_like(cur_pre)
    #                 gt = np.zeros_like(cur_tar)
    #                 pred[cur_pre == c] = 1
    #                 gt[cur_tar == c] = 1
    #                 if pred.sum() > 0 and gt.sum() > 0 and c != self.ignore_index:
    #                     cur_hd = hd95(pred, gt, voxelspacing=dict_spacing['img' + uniq_case[i]])
    #                     # cur_hd = np.random.randint(low=0, high=100, size=1)[0]
    #                 # elif pred.sum() == 0 and gt.sum() == 0:
    #                 else:
    #                     cur_hd = np.nan
    #                 # else:
    #                 #     inf = np.sum(np.array(cur_pre.shape) * np.array(dict_spacing['img' + uniq_case[i]]))
    #                 #     cur_hd = inf
    #                 per_hd_metrics.append(cur_hd)
    #
    #                 hd_elapsed = hd_time.since_start()
    #                 hd_percent = (c + 1) / float(num_classes)
    #                 hd_eta = int(hd_elapsed * (1 - hd_percent) / hd_percent + 0.5)
    #                 hd_elapsed = str(datetime.timedelta(seconds=int(hd_elapsed)))
    #                 hd_eta = str(datetime.timedelta(seconds=int(hd_eta)))
    #                 print_log(f'Compute HD per class [{c + 1}/{num_classes}] '
    #                           f'elapsed: {hd_elapsed} eta: {hd_eta} '
    #                           f'class name: {class_names[c]} '
    #                           f'HD: {cur_hd:.2f}',
    #                           logger)
    #             hd_metrics.append(per_hd_metrics)
    #
    #         case_elapsed = case_time.since_start()
    #         case_percent = (i + 1) / float(num_cases)
    #         case_eta = int(case_elapsed * (1 - case_percent) / case_percent + 0.5)
    #         case_elapsed = str(datetime.timedelta(seconds=int(case_elapsed)))
    #         case_eta = str(datetime.timedelta(seconds=int(case_eta)))
    #         print_case = f'Compute metrics per case [{i + 1}/{num_cases}] ' \
    #                      f'elapsed: {case_elapsed} eta: {case_eta} '
    #         print_log(print_case, logger)
    #
    #     return list_metrics, hd_metrics
    @staticmethod
    def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
                            num_classes: int, ignore_index: int):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """

        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_pred_label = torch.histc(
            pred_label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_label = torch.histc(
            label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label

    @staticmethod
    def total_area_to_metrics(total_area_intersect: np.ndarray,
                              total_area_union: np.ndarray,
                              total_area_pred_label: np.ndarray,
                              total_area_label: np.ndarray,
                              metrics: List[str] = ['mIoU'],
                              nan_to_num: Optional[int] = None,
                              beta: int = 1):
        """Calculate evaluation metrics
        Args:
            total_area_intersect (np.ndarray): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (np.ndarray): The union of prediction and ground
                truth histogram on all classes.
            total_area_pred_label (np.ndarray): The prediction histogram on
                all classes.
            total_area_label (np.ndarray): The ground truth histogram on
                all classes.
            metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
                'mDice'.
            nan_to_num (int, optional): If specified, NaN values will be
                replaced by the numbers defined by the user. Default: None.
            beta (int): Determines the weight of recall in the combined score.
                Default: 1.
        Returns:
            Dict[str, np.ndarray]: per category evaluation metrics,
                shape (num_classes, ).
        """

        def f_score(precision, recall, beta=1):
            """calculate the f-score value.

            Args:
                precision (float | torch.Tensor): The precision value.
                recall (float | torch.Tensor): The recall value.
                beta (int): Determines the weight of recall in the combined
                    score. Default: 1.

            Returns:
                [torch.tensor]: The f-score value.
            """
            score = (1 + beta ** 2) * (precision * recall) / (
                    (beta ** 2 * precision) + recall)
            return score

        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f'metrics {metrics} is not supported')

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({'aAcc': all_acc})
        for metric in metrics:
            if metric == 'mIoU':
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics['IoU'] = iou
                ret_metrics['Acc'] = acc
            elif metric == 'mDice':
                dice = 2 * total_area_intersect / (
                        total_area_pred_label + total_area_label)
                acc = total_area_intersect / total_area_label
                ret_metrics['Dice'] = dice
                ret_metrics['Acc'] = acc
            elif metric == 'mFscore':
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor([
                    f_score(x[0], x[1], beta) for x in zip(precision, recall)
                ])
                ret_metrics['Fscore'] = f_value
                ret_metrics['Precision'] = precision
                ret_metrics['Recall'] = recall
        ret_metrics = {
            metric: value.numpy()
            for metric, value in ret_metrics.items()
        }
        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })
        return ret_metrics

    @staticmethod
    def hd_to_metrics(prediction: List[np.ndarray], target: List[np.ndarray]) -> List:
        try:
            from medpy.metric.binary import hd95
        except ImportError:
            warnings.warn("medpy not installed, cannot compute hd_distance")
            return list()
        result_list = []
        for cur_pre, cur_tar in zip(prediction, target):
            cur_hd = hd95(cur_pre, cur_tar)
            result_list.append(cur_hd)

        return result_list

def _to_cpu(data: Any) -> Any:
    """transfer all tensors and BaseDataElement to cpu."""
    if isinstance(data, (Tensor, BaseDataElement)):
        return data.to('cpu')
    elif isinstance(data, list):
        return [_to_cpu(d) for d in data]
    elif isinstance(data, tuple):
        return tuple(_to_cpu(d) for d in data)
    elif isinstance(data, dict):
        return {k: _to_cpu(v) for k, v in data.items()}
    else:
        return data