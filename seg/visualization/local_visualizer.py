# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, List, Optional, Tuple

import mmcv
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.dist import master_only
from mmengine.structures import PixelData
from mmengine.visualization import Visualizer
from mmengine.visualization.utils import convert_overlay_heatmap, img_from_canvas

from seg.registry import VISUALIZERS
from mmseg.visualization import SegLocalVisualizer as _SegLocalVisualizer
from mmseg.structures import SegDataSample
from mmseg.utils import get_classes, get_palette

@VISUALIZERS.register_module()
class SegLocalVisualizer(_SegLocalVisualizer):
    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 classes: Optional[List] = None,
                 palette: Optional[List] = None,
                 dataset_name: Optional[str] = None,
                 alpha: float = 0.8,
                 **kwargs):
        super().__init__(name, image, vis_backends, save_dir, **kwargs)
        self.alpha: float = alpha
        self.set_dataset_meta(palette, classes, dataset_name)
    @master_only
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            data_sample: Optional[SegDataSample] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. it is usually used when the display
        is not available.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            gt_sample (:obj:`SegDataSample`, optional): GT SegDataSample.
                Defaults to None.
            pred_sample (:obj:`SegDataSample`, optional): Prediction
                SegDataSample. Defaults to None.
            draw_gt (bool): Whether to draw GT SegDataSample. Default to True.
            draw_pred (bool): Whether to draw Prediction SegDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
        """
        classes = self.dataset_meta.get('classes', None)
        palette = self.dataset_meta.get('palette', None)

        gt_img_data = None
        pred_img_data = None

        if draw_gt and data_sample is not None and 'gt_sem_seg' in data_sample:
            gt_img_data = image
            assert classes is not None, 'class information is ' \
                                        'not provided when ' \
                                        'visualizing semantic ' \
                                        'segmentation results.'
            gt_img_data = self._draw_sem_seg(gt_img_data,
                                             data_sample.gt_sem_seg, classes,
                                             palette)

        if (draw_pred and data_sample is not None
                and 'pred_sem_seg' in data_sample):
            pred_img_data = image
            assert classes is not None, 'class information is ' \
                                        'not provided when ' \
                                        'visualizing semantic ' \
                                        'segmentation results.'
            pred_img_data = self._draw_sem_seg(pred_img_data,
                                               data_sample.pred_sem_seg,
                                               classes, palette)

        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        else:
            drawn_img = pred_img_data

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(mmcv.bgr2rgb(drawn_img), out_file)
        else:
            self.add_image(name, drawn_img, step)

    # def _draw_sem_seg(self, image: np.ndarray, sem_seg: PixelData,
    #                   classes: Optional[List],
    #                   palette: Optional[List]) -> np.ndarray:
    #     """Draw semantic seg of GT or prediction.
    #
    #     Args:
    #         image (np.ndarray): The image to draw.
    #         sem_seg (:obj:`PixelData`): Data structure for pixel-level
    #             annotations or predictions.
    #         classes (list, optional): Input classes for result rendering, as
    #             the prediction of segmentation model is a segment map with
    #             label indices, `classes` is a list which includes items
    #             responding to the label indices. If classes is not defined,
    #             visualizer will take `cityscapes` classes by default.
    #             Defaults to None.
    #         palette (list, optional): Input palette for result rendering, which
    #             is a list of color palette responding to the classes.
    #             Defaults to None.
    #
    #     Returns:
    #         np.ndarray: the drawn image which channel is RGB.
    #     """
    #     num_classes = len(classes)
    #
    #     sem_seg = sem_seg.cpu().data
    #     ids = np.unique(sem_seg)[::-1]
    #     ids = ids[ids > 0]
    #     if ids.size != 0:
    #         legal_indices = ids < num_classes
    #         ids = ids[legal_indices]
    #         labels = np.array(ids, dtype=np.int64)
    #
    #         colors = [palette[label] for label in labels]
    #
    #         mask = np.zeros_like(image, dtype=np.uint8)
    #         for label, color in zip(labels, colors):
    #             mask[sem_seg[0] == label, :] = color
    #         for i in range(mask.shape[0]):
    #             mask_i = mask[i, i]
    #             if np.all(mask_i == 0):
    #                 mask[i, i] = image[i, i]
    #         color_seg = mask.astype(np.uint8)
    #         # color_seg = (image * (1 - self.alpha) + mask * self.alpha).astype(
    #         #     np.uint8)
    #         # for idx in np.where(mask != 0):
    #         #     image[idx] = mask[idx]
    #         # color_seg = (mask + image * (mask == [0, 0, 0])).astype(
    #         #     np.uint8)
    #     else:
    #         color_seg = image.astype(np.uint8)
    #     self.set_image(color_seg)
    #     return color_seg

    def _draw_sem_seg(self, image: np.ndarray, sem_seg: PixelData,
                      classes: Optional[List],
                      palette: Optional[List]) -> np.ndarray:
        """Draw semantic seg of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            sem_seg (:obj:`PixelData`): Data structure for pixel-level
                annotations or predictions.
            classes (list, optional): Input classes for result rendering, as
                the prediction of segmentation model is a segment map with
                label indices, `classes` is a list which includes items
                responding to the label indices. If classes is not defined,
                visualizer will take `cityscapes` classes by default.
                Defaults to None.
            palette (list, optional): Input palette for result rendering, which
                is a list of color palette responding to the classes.
                Defaults to None.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        num_classes = len(classes)

        sem_seg = sem_seg.cpu().data
        ids = np.unique(sem_seg)[::-1]
        legal_indices = ids < num_classes
        ids = ids[legal_indices]
        labels = np.array(ids, dtype=np.int64)

        colors = [palette[label] for label in labels]

        mask = np.zeros_like(image, dtype=np.uint8)
        img_mask = np.ones_like(image, dtype=np.uint8)
        for label, color in zip(labels, colors):
            mask[sem_seg[0] == label, :] = color
            if label != 0:
                img_mask[sem_seg[0] == label, :] = [0, 0, 0]

        color_seg = (image * img_mask + mask).astype(
            np.uint8)
        self.set_image(color_seg)
        return color_seg