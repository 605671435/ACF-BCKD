# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Optional, Sequence, Dict

from PIL import Image
import SimpleITK as sitk
import os
from mmengine.logging import MMLogger, print_log
import numpy as np
import torch

import mmcv
import mmengine.fileio as fileio

from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.dist import master_only
# from seg.registry import HOOKS
from mmseg.structures import SegDataSample
from seg.visualization import SegLocalVisualizer
from mmengine.visualization.visualizer import Visualizer
def arr_to_img(arr):
    min = np.amin(arr)
    max = np.amax(arr)
    new = (arr - min) * (1. / (max - min)) * 255
    new = new.astype(np.uint8)
    # img = Image.fromarray(new)
    return np.rot90(new)

# @HOOKS.register_module()
class SegVisualizationHook(Hook):
    """Segmentation Visualization Hook. Used to visualize validation and
    testing process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.

        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.

    """

    def __init__(self,
                 draw_table: bool = False,
                 draw_ct: bool = False,
                 draw: bool = False,
                 save_featmap = False,
                 interval: int = 50,
                 show: bool = False,
                 wait_time: float = 0.,
                 backend_args: Optional[dict] = None,
                 k_fold: bool = True):

        self._visualizer: SegLocalVisualizer = \
            SegLocalVisualizer.get_current_instance()
        self.interval = interval
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.backend_args = backend_args.copy() if backend_args else None

        self.draw = draw
        self.draw_table = draw_table
        self.draw_ct = draw_ct
        self.save_featmap = save_featmap
        self.k_fold = k_fold

        if save_featmap is True:
            if not os.path.exists('./work_dirs/featmaps/tmp'):
                os.makedirs('./work_dirs/featmaps/tmp')
        assert draw_ct is not True or draw_table is not True, \
        'assert error'

    @master_only
    def _after_iter(self,
                    runner: Runner,
                    batch_idx: int,
                    data_batch: dict,
                    outputs: Sequence[SegDataSample],
                    mode: str = 'val') -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): Outputs from model.
            mode (str): mode (str): Current mode of runner. Defaults to 'val'.
        """
        if self.save_featmap is True:
            if self.every_n_inner_iters(runner.iter, self.interval):
                os.rename('./work_dirs/featmaps/tmp', './work_dirs/featmaps/iter_' + str(runner.iter))
        if self.draw is False or mode == 'train':
            return

        if self.every_n_inner_iters(batch_idx, self.interval):
            for output in outputs:
                img_path = output.img_path
                img_bytes = fileio.get(
                    img_path, backend_args=self.backend_args)
                img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                window_name = f'{mode}_{osp.basename(img_path)}'

                self._visualizer.add_datasample(
                    window_name,
                    img,
                    data_sample=output,
                    show=self.show,
                    wait_time=self.wait_time,
                    step=runner.iter)

    def compute_metric(self, pred_sem_seg, gt_sem_seg, num_classes, metric='Dice'):
        intersect = pred_sem_seg[pred_sem_seg == gt_sem_seg]
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_pred_label = torch.histc(
            pred_sem_seg.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_label = torch.histc(
            gt_sem_seg.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()

        if metric == 'mIoU':
            area_union = area_pred_label + area_label - area_intersect
            iou = area_intersect / area_union

            return np.round(iou.numpy() * 100, 2)

        dice = 2 * area_intersect / (
                area_pred_label + area_label)
        acc = area_intersect / area_label

        return np.round(dice.numpy() * 100, 2), np.round(acc.numpy() * 100, 2)

    @master_only
    def before_run(self, runner) -> None:
        # self._visualizer.add_config(runner.cfg)
        # for vis_backend in runner.visualizer._vis_backends.values():
        #     vis_backend._init_env()
        #     vis_backend.add_config(runner.cfg)
        if self.draw_table is True or self.draw_ct is True:

            vis_backend = runner.visualizer._vis_backends.get('WandbVisBackend')
            # vis_backend._init_env()
            self.wandb = vis_backend.experiment

            logger: MMLogger = MMLogger.get_current_instance()

            if hasattr(runner.train_dataloader.dataset, 'METAINFO'):
                self.class_info = runner.train_dataloader.dataset.METAINFO['classes']
            else:
                self.class_info = runner.train_dataloader.dataset.metainfo['classes']

            self.num_classes = len(self.class_info)

            if self.draw_table is True:
                columns = ["id", "iter", "image", "gt", "pred"]
                columns.extend(["%_" + c for c in self.class_info])
                print_log('Create a wandb table.', logger)
                self.test_table = self.wandb.Table(columns=columns)

                class_id = list(range(self.num_classes))
                self.class_set = self.wandb.Classes([{'name': name, 'id': id}
                                           for name, id in zip(self.class_info, class_id)])
            if self.draw_ct is True:
                self.pred_path = os.path.join(self.wandb.run.dir, 'pred_data')
                if not os.path.exists(self.pred_path):
                    os.makedirs(self.pred_path)
                columns = ["id", "iter", "gt", "pred", "dice", "acc"]
                print_log('Create a wandb table.', logger)
                self.test_table = self.wandb.Table(columns=columns)

    def add_data(self,
                 runner,
                 batch_idx: int,
                 data_batch: dict,
                 outputs):
        if self.every_n_inner_iters(batch_idx, self.interval):
            for output in outputs:
                metric = runner.val_evaluator.metrics[0].metrics[0]
                if self.draw_ct is True:
                    img_path = output.img_path.split('/')[-1].split('.')[0]

                    gt_sem_seg_arr = arr_to_img(output.gt_sem_seg_3d.data[0].cpu().permute(1, 2, 0).numpy().astype('uint8'))
                    gt_gif = [Image.fromarray(frame) for frame in gt_sem_seg_arr]
                    gt_gif_filename = os.path.join(runner.work_dir, 'gt_tmp.gif')
                    gt_gif[0].save(gt_gif_filename, save_all=True, append_images=gt_gif, duration=0.1)

                    pred_sem_seg = output.pred_sem_seg_3d.data[0].cpu().numpy().astype('uint8')
                    pred_nii_file = sitk.GetImageFromArray(pred_sem_seg)
                    sitk.WriteImage(pred_nii_file, os.path.join(self.pred_path, (img_path + '.nii.gz')))

                    pred_sem_seg_arr = arr_to_img(pred_sem_seg.transpose(1, 2, 0))
                    pred_gif = [Image.fromarray(frame) for frame in pred_sem_seg_arr]
                    pred_gif_filename = os.path.join(runner.work_dir, 'pred_tmp.gif')
                    pred_gif[0].save(pred_gif_filename, save_all=True, append_images=pred_gif, duration=0.1)

                    dice, acc = self.compute_metric(outputs[0].pred_sem_seg_3d.data.squeeze(),
                                                    outputs[0].gt_sem_seg_3d.data.squeeze(),
                                                    self.num_classes)
                    dices = []
                    accs = []
                    form = "{}:{:.2f}, "
                    metrics_pos = ""
                    for i in range(self.num_classes):
                        dices.append(self.class_info[i])
                        dices.append(dice[i])
                        accs.append(self.class_info[i])
                        accs.append(acc[i])
                        metrics_pos += form
                    metrics_pos = "".join(list(metrics_pos)[:-2])
                    logger: MMLogger = MMLogger.get_current_instance()
                    print_log('Add data to wandb table.', logger)
                    self.test_table.add_data(img_path,
                                             runner.iter,
                                             self.wandb.Video(gt_gif_filename),
                                             self.wandb.Video(pred_gif_filename),
                                             metrics_pos.format(*dices),
                                             metrics_pos.format(*accs))
                elif self.draw_table is True:
                    img_path = output.img_path.split('/')[-1].split('.')[0]
                    ori_img = data_batch['inputs'][0].permute(1, 2, 0).cpu().numpy().astype('uint8')

                    gt_sem_seg = output.gt_sem_seg.data.cpu().permute(1, 2, 0).numpy().astype('uint8')
                    gt_sem_seg = mmcv.imresize(gt_sem_seg[..., -1],
                                               (ori_img.shape[1], ori_img.shape[0]),
                                               interpolation='nearest',
                                               backend='pillow')
                    pred_sem_seg = output.pred_sem_seg.data.cpu().permute(1, 2, 0).numpy().astype('uint8')
                    pred_sem_seg = mmcv.imresize(pred_sem_seg[..., -1],
                                                 (ori_img.shape[1], ori_img.shape[0]),
                                                 interpolation='nearest',
                                                 backend='pillow')

                    if metric == 'mIoU':
                        metric = self.compute_metric(outputs[0].pred_sem_seg.data,
                                                    outputs[0].gt_sem_seg.data,
                                                    self.num_classes,
                                                    metric)
                    else:
                        metric, acc = self.compute_metric(outputs[0].pred_sem_seg.data,
                                                        outputs[0].gt_sem_seg.data,
                                                        self.num_classes)

                    annotated = self.wandb.Image(ori_img, classes=self.class_set,
                                            masks={"ground_truth": {"mask_data": gt_sem_seg}})
                    predicted = self.wandb.Image(ori_img, classes=self.class_set,
                                            masks={"predicted_mask": {"mask_data": pred_sem_seg}})
                    logger: MMLogger = MMLogger.get_current_instance()
                    print_log('Add data to wandb table.', logger)

                    row = [img_path, runner.iter, self.wandb.Image(ori_img), annotated, predicted]
                    row.extend(metric)

                    self.test_table.add_data(*row)

    @master_only
    def after_val_iter(self,
                       runner,
                       batch_idx: int,
                       data_batch: dict,
                       outputs) -> None:
        if self.draw_table is True or self.draw_ct is True:
            self.add_data(runner,
                     batch_idx,
                     data_batch,
                     outputs)

    @master_only
    def after_test_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: dict,
                        outputs) -> None:
        if self.draw_table is True or self.draw_ct is True:
            self.add_data(runner,
                     batch_idx,
                     data_batch,
                     outputs)

    @master_only
    def after_run(self, runner) -> None:
        if self.draw_table is True or self.draw_ct is True:
            logger: MMLogger = MMLogger.get_current_instance()
            print_log('Log table to wandb.', logger)
            self.wandb.log({"test_predictions": self.test_table})
        # vis_backends = runner.visualizer._vis_backends
        # for vis_backend in vis_backends.values():
        #     vis_backend._env_initialized = False

        # for vis_backend in runner.visualizer.vis_backends:
        #     if vis_backend['type'] == 'WandbVisBackend':
        #         vis_backend.finish
        # runner.visualizer.close()
        # vis_backend = runner.visualizer._vis_backends.get('WandbVisBackend')
        # if vis_backend is not None:
        # # vis_backend._init_env()
        #     self.wandb = vis_backend.experiment
        #     self.wandb.finish()
        # # del runner.visualizer
        # # from mmengine.registry import VISUALIZERS
        # from mmengine.visualization import Visualizer
        # from seg.visualization.local_visualizer import SegLocalVisualizer
        # Visualizer._instance_dict.clear()
        # SegLocalVisualizer._instance_dict.clear()
