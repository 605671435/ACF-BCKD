# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image
from prettytable import PrettyTable

from seg.registry import METRICS


@METRICS.register_module()
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
                 iou_metrics: List[str] = ['mIoU'],
                 hd_metric: bool = False,
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.metrics = iou_metrics
        self.hd_metric = hd_metric
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
        num_classes = len(self.dataset_meta['classes'])

        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample['gt_sem_seg']['data'].squeeze().to(
                    pred_label)
                self.results.append((pred_label, label))
            # format_result
            if self.output_dir is not None:
                basename = osp.splitext(osp.basename(
                    data_sample['img_path']))[0]
                png_filename = osp.abspath(
                    osp.join(self.output_dir, f'{basename}.png'))
                output_mask = pred_label.cpu().numpy()
                # The index range of official ADE20k dataset is from 0 to 150.
                # But the index range of output is from 0 to 149.
                # That is because we set reduce_zero_label=True.
                if data_sample.get('reduce_zero_label', False):
                    output_mask = output_mask + 1
                output = Image.fromarray(output_mask.astype(np.uint8))
                output.save(png_filename)

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

        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        results = tuple(zip(*results))
        class_names = self.dataset_meta['classes']
        num_classes = len(class_names)

        total_pred_label = torch.stack(results[0], dim=0).numpy()
        total_label = torch.stack(results[1], dim=0).numpy()
        ret_metrics = self.label_to_metrics(total_pred_label,
                                            total_label,
                                            num_classes,
                                            self.ignore_index)
        # summary table
        ret_metrics_summary = OrderedDict({
            'Dice': np.round(np.nanmean(ret_metrics['Dice']) * 100, 2)
        })
        ret_metrics_summary['HD95'] = np.round(np.nanmean(ret_metrics['HD95']), 2)
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                metrics[key] = val
            else:
                metrics['m' + key] = val

        # each class table
        ret_metrics_class = OrderedDict({
            'Dice': np.round(ret_metrics['Dice'] * 100, 2)
        })
        ret_metrics_class['HD95'] = np.round(ret_metrics['HD95'], 2)
        class_names = class_names[1:]

        for metric in self.metrics:
            metric = metric.split('m')[-1] if metric.startswith('m') else metric
            for class_key, metric_value in zip(class_names, ret_metrics_class[metric]):
                metrics[f'{metric} ({class_key})'] = np.round(metric_value.astype('float64'), 4)

        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics

    @staticmethod
    def label_to_metrics(prediction: List[np.ndarray], target: List[np.ndarray],
                         num_classes: int, ignore_index: int):
        try:
            from medpy import metric
        except ImportError:
            warnings.warn("medpy not installed, cannot compute hd_distance")
            return list()

        def calculate_metric_percase(pred, gt):
            pred[pred > 0] = 1
            gt[gt > 0] = 1
            if pred.sum() > 0 and gt.sum() > 0:
                dice = metric.binary.dc(pred, gt)
                hd95 = metric.binary.hd95(pred, gt)
                # hd95 = np.random.randint(low=0, high=100, size=1)[0]
                return dice, hd95
            elif pred.sum() > 0 and gt.sum() == 0:
                return 1, 0
            else:
                return 0, 0

        dice_list = []
        hd95_list = []
        for c in range(num_classes):
            if c == ignore_index:
                continue
            else:
                dice, hd95 = calculate_metric_percase(prediction == c, target == c)
                dice_list.append(dice)
                hd95_list.append(hd95)
        ret_metrics = OrderedDict({'Dice': np.array(dice_list)})
        ret_metrics['HD95'] = np.array(hd95_list)
        return ret_metrics
