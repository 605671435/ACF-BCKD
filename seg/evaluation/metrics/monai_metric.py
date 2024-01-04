# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional
import logging
import torch
from mmengine.dist import (broadcast_object_list, collect_results,
                           is_main_process)
from mmengine.evaluator import BaseMetric
from mmengine.evaluator.metric import _to_cpu
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
import nibabel as nib
from prettytable import PrettyTable
from .confusion_matrix import *
from monai.transforms import AsDiscrete,  Activations
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
from monai.data import decollate_batch
from monai.utils import MetricReduction
from utils.utils import resample_3d


class HD95Metric(HausdorffDistanceMetric):

    def __init__(self, **kwargs):
        super().__init__(percentile=95, **kwargs)


mapping = dict(
    Dice=DiceMetric,
    IoU=MeanIoU,
    HD95=HD95Metric)


class MonaiMetric(BaseMetric):
    def __init__(self,
                 metrics: List[str],
                 num_classes: int,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = './preds',
                 save_outputs: bool = False,
                 include_background: bool = False,
                 get_not_nans: bool = True,
                 one_hot: bool = True,
                 print_per_class : bool = True,
                 reduction: str = 'mean',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.output_dir = output_dir
        self.save_outputs = save_outputs
        if self.save_outputs:
            mkdir_or_exist(self.output_dir)
        self.print_per_class = print_per_class
        self.include_background = include_background
        self.one_hot = one_hot
        if self.one_hot:
            self.post_label = AsDiscrete(to_onehot=num_classes)
            self.post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
        else:
            self.post_sigmoid = Activations(sigmoid=True)
            self.post_pred = AsDiscrete(argmax=False, threshold=0.5)
        # self.num_classes = num_classes if include_background else num_classes - 1
        self.metric_names = metrics
        self.metric_funcs = []
        self.ret_metrics_class_pre_case = []
        for metric in metrics:
            metric = mapping[metric](
                include_background=include_background,
                reduction=reduction,
                get_not_nans=get_not_nans)
            self.metric_funcs.append(metric)

    def process(self, data_batch: torch.Tensor, data_samples: dict) -> None:
        target = data_samples['label'].cuda()
        if not data_batch.is_cuda:
            target = target.cpu()
        if self.save_outputs:
            if self.one_hot:
                self.save_predictions(data_batch, data_samples, self.output_dir)
            else:
                self.save_brats23(data_batch, data_samples, self.output_dir)
            return
        val_labels_list = decollate_batch(target)
        val_outputs_list = decollate_batch(data_batch)
        if self.one_hot:
            val_labels_convert = [self.post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_output_convert = [self.post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
        else:
            val_labels_convert = val_labels_list
            val_output_convert = [
                self.post_pred(self.post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]

        results_list = dict()
        for name, metric_func in zip(self.metric_names, self.metric_funcs):
            metric_func.reset()
            metric_func(y_pred=val_output_convert, y=val_labels_convert)
            # results, not_nans = metric_func.aggregate()
            results = metric_func.get_buffer()
            results_list[name] = results[0].detach().cpu().numpy()
        ret_metrics = self.format_metrics(results_list)
        self.results.append(ret_metrics)

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
        if self.save_outputs:
            return {}
        metrics = dict()
        for key in results[0].keys():
            metrics[key] = np.round(np.nanmean([case_metric[key] for case_metric in results]), 2)

        return metrics

    def format_metrics(self, ret_metrics: dict) -> dict:
        logger: MMLogger = MMLogger.get_current_instance()

        if self.include_background:
            class_names = self.dataset_meta['classes']
        else:
            class_names = self.dataset_meta['classes'][1:]

        ret_metrics_summary = OrderedDict({
            name: np.nanmean(ret_metrics[name])
            for name in self.metric_names})

        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == 'HD95':
                ret_metrics_summary[key] = np.round(val * 1, 2)
            else:
                ret_metrics_summary[key] = np.round(val * 100, 2)
            metrics[key] = ret_metrics_summary[key]

        # each class table
        ret_metrics_class = OrderedDict()
        for name in self.metric_names:
            if name == 'HD95':
                ret_metrics_class[name] = np.round(ret_metrics[name] * 1, 2)
            else:
                ret_metrics_class[name] = np.round(ret_metrics[name] * 100, 2)
            ret_metrics_class[name] = np.round(np.append(ret_metrics_class[name], ret_metrics_summary[name]), 2)
        for name in self.metric_names:  # ['Dice', 'Jaccard', 'HD95']
            for class_key, metric_value in zip(class_names, ret_metrics_class[name]):
                metrics[f'{name} ({class_key})'] = metric_value

        # self.ret_metrics_class_pre_case.append(ret_metrics_class)
        ret_metrics_class.update({'Class': class_names + ('Average',)})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        if self.print_per_class:
            print_log('per class results:', logger)
            print_log('\n' + class_table_data.get_string(), logger=logger)
        return metrics

    @staticmethod
    def save_predictions(outputs: torch.Tensor, data_samples: dict, output_dir: str) -> None:
        outputs = torch.softmax(outputs, 1)
        outputs = torch.argmax(outputs, 1)

        img_path = data_samples['label_meta_dict']['filename_or_obj'][0]
        shape = data_samples['label_meta_dict']['spatial_shape'][0]
        affine = data_samples['label_meta_dict']['original_affine'][0]

        outputs = outputs.cpu().numpy().astype(np.uint8)[0]
        val_outputs = resample_3d(outputs, shape)

        save_dirs = os.path.join(
            output_dir,
            osp.basename(img_path.replace('label', 'pred')))
        nib.save(
            nib.Nifti1Image(val_outputs.astype(np.uint8), affine),
            save_dirs)
        logger: MMLogger = MMLogger.get_current_instance()
        print_log('Prediction is saved at:', logger=logger)
        print_log(save_dirs, logger=logger)

    @staticmethod
    def save_brats23(outputs: torch.Tensor, data_samples: dict, output_dir: str) -> None:
        img_path = data_samples['label_meta_dict']['filename_or_obj'][0]
        save_dirs = os.path.join(
            output_dir,
            osp.basename(img_path.replace('label', 'pred')))
        shape = data_samples['label_meta_dict']['spatial_shape'][0]
        affine = data_samples['label_meta_dict']['original_affine'][0]
        prob = torch.sigmoid(outputs)
        seg = prob[0].detach().cpu().numpy()
        seg = (seg > 0.5).astype(np.int8)
        seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
        seg_out[seg[1] == 1] = 2
        seg_out[seg[0] == 1] = 1
        seg_out[seg[2] == 1] = 3
        nib.save(nib.Nifti1Image(seg_out.astype(np.uint8), affine), save_dirs)

if __name__ == '__main__':
    evaluator = MonaiMetric(
        metrics=['Dice', 'IoU', 'HD95'],
        num_classes=14)

