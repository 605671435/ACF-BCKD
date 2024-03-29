# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import List, Optional, Sequence, Union

import mmcv
import mmengine
import numpy as np
import torch
import torch.nn as nn
from mmcv.transforms import Compose
from mmengine.infer.infer import BaseInferencer, ModelType
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.config import Config
from mmengine.runner.checkpoint import (_load_checkpoint,
                                        _load_checkpoint_to_model)
from PIL import Image

from seg.registry import MODELS

from mmseg.structures import SegDataSample
from mmseg.utils import ConfigType, SampleList, get_classes, get_palette
from seg.visualization import SegLocalVisualizer
from mmseg.visualization import SegLocalVisualizer as MMSegLocalVisualizer

InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[SegDataSample, SampleList]


class MMSegInferencer(BaseInferencer):
    """Semantic segmentation inferencer, provides inference and visualization
    interfaces. Note: MMEngine >= 0.5.0 is required.

    Args:
        model (str, optional): Path to the config file or the model name
            defined in metafile. Take the `mmseg metafile <https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/fcn/fcn.yml>`_
            as an example the `model` could be
            "fcn_r50-d8_4xb2-40k_cityscapes-512x1024", and the weights of model
            will be download automatically. If use config file, like
            "configs/fcn/fcn_r50-d8_4xb2-40k_cityscapes-512x1024.py", the
            `weights` should be defined.
        weights (str, optional): Path to the checkpoint. If it is not specified
            and model is a model name of metafile, the weights will be loaded
            from metafile. Defaults to None.
        classes (list, optional): Input classes for result rendering, as the
            prediction of segmentation model is a segment map with label
            indices, `classes` is a list which includes items responding to the
            label indices. If classes is not defined, visualizer will take
            `cityscapes` classes by default. Defaults to None.
        palette (list, optional): Input palette for result rendering, which is
            a list of color palette responding to the classes. If palette is
            not defined, visualizer will take `cityscapes` palette by default.
            Defaults to None.
        dataset_name (str, optional): `Dataset name or alias <https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/mmseg/utils/class_names.py#L302-L317>`_
            visulizer will use the meta information of the dataset i.e. classes
            and palette, but the `classes` and `palette` have higher priority.
            Defaults to None.
        device (str, optional): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.
        scope (str, optional): The scope of the model. Defaults to 'mmseg'.
    """ # noqa

    preprocess_kwargs: set = set()
    forward_kwargs: set = {'mode', 'out_dir'}
    visualize_kwargs: set = {'show', 'wait_time', 'img_out_dir', 'opacity'}
    postprocess_kwargs: set = {'pred_out_dir', 'return_datasample'}

    def __init__(self,
                 model: Union[ModelType, str],
                 weights: Optional[str] = None,
                 classes: Optional[Union[str, List]] = None,
                 palette: Optional[Union[str, List]] = None,
                 dataset_name: Optional[str] = None,
                 device: Optional[str] = None,
                 scope: Optional[str] = 'seg') -> None:
        # A global counter tracking the number of images processes, for
        # naming of the output images
        self.num_visualized_imgs = 0
        self.num_pred_imgs = 0
        init_default_scope(scope if scope else 'seg')
        super().__init__(
            model=model, weights=weights, device=device, scope=scope)

        if device == 'cpu' or not torch.cuda.is_available():
            self.model = revert_sync_batchnorm(self.model)

        assert isinstance(self.visualizer, SegLocalVisualizer) or isinstance(self.visualizer, MMSegLocalVisualizer)
        self.visualizer.set_dataset_meta(palette, classes, dataset_name)

    def _init_model(
        self,
        cfg: ConfigType,
        weights: Optional[str],
        device: str = 'cpu',
    ) -> nn.Module:
        """Initialize the model with the given config and checkpoint on the
        specific device.

        Args:
            cfg (ConfigType): Config containing the model information.
            weights (str, optional): Path to the checkpoint.
            device (str, optional): Device to run inference. Defaults to 'cpu'.

        Returns:
            nn.Module: Model loaded with checkpoint.
        """
        checkpoint: Optional[dict] = None
        if weights is not None:
            checkpoint = _load_checkpoint(weights, map_location='cpu')

        if not cfg:
            assert checkpoint is not None
            try:
                # Prefer to get config from `message_hub` since `message_hub`
                # is a more stable module to store all runtime information.
                # However, the early version of MMEngine will not save config
                # in `message_hub`, so we will try to load config from `meta`.
                cfg_string = checkpoint['message_hub']['runtime_info']['cfg']
            except KeyError:
                assert 'meta' in checkpoint, (
                    'If model(config) is not provided, the checkpoint must'
                    'contain the config string in `meta` or `message_hub`, '
                    'but both `meta` and `message_hub` are not found in the '
                    'checkpoint.')
                meta = checkpoint['meta']
                if 'cfg' in meta:
                    cfg_string = meta['cfg']
                else:
                    raise ValueError(
                        'Cannot find the config in the checkpoint.')
            cfg.update(
                Config.fromstring(cfg_string, file_format='.py')._cfg_dict)

        # Delete the `pretrained` field to prevent model from loading the
        # the pretrained weights unnecessarily.
        if cfg.model.get('pretrained') is not None:
            del cfg.model.pretrained

        model = MODELS.build(cfg.model)
        model.cfg = cfg
        self._load_weights_to_model(model, checkpoint, cfg)
        model.to(device)
        model.eval()
        return model

    def _load_weights_to_model(self, model: nn.Module,
                               checkpoint: Optional[dict],
                               cfg: Optional[ConfigType]) -> None:
        """Loading model weights and meta information from cfg and checkpoint.

        Subclasses could override this method to load extra meta information
        from ``checkpoint`` and ``cfg`` to model.

        Args:
            model (nn.Module): Model to load weights and meta information.
            checkpoint (dict, optional): The loaded checkpoint.
            cfg (Config or ConfigDict, optional): The loaded config.
        """

        if checkpoint is not None:
            _load_checkpoint_to_model(model, checkpoint)
            checkpoint_meta = checkpoint.get('meta', {})
            # save the dataset_meta in the model for convenience
            if 'dataset_meta' in checkpoint_meta:
                # mmsegmentation 1.x
                model.dataset_meta = {
                    'classes': checkpoint_meta['dataset_meta'].get('classes'),
                    'palette': checkpoint_meta['dataset_meta'].get('palette')
                }
            elif 'CLASSES' in checkpoint_meta:
                # mmsegmentation 0.x
                classes = checkpoint_meta['CLASSES']
                palette = checkpoint_meta.get('PALETTE', None)
                model.dataset_meta = {'classes': classes, 'palette': palette}
            else:
                warnings.warn(
                    'dataset_meta or class names are not saved in the '
                    'checkpoint\'s meta data, use classes of Cityscapes by '
                    'default.')
                model.dataset_meta = {
                    'classes': get_classes('cityscapes'),
                    'palette': get_palette('cityscapes')
                }
        else:
            warnings.warn('Checkpoint is not loaded, and the inference '
                          'result is calculated by the randomly initialized '
                          'model!')
            warnings.warn(
                'weights is None, use cityscapes classes by default.')
            model.dataset_meta = {
                'classes': get_classes('cityscapes'),
                'palette': get_palette('cityscapes')
            }

    def __call__(self,
                 inputs: InputsType,
                 return_datasamples: bool = False,
                 batch_size: int = 1,
                 show: bool = False,
                 wait_time: int = 0,
                 out_dir: str = '',
                 img_out_dir: str = 'vis',
                 pred_out_dir: str = 'pred',
                 **kwargs) -> dict:
        """Call the inferencer.

        Args:
            inputs (Union[list, str, np.ndarray]): Inputs for the inferencer.
            return_datasamples (bool): Whether to return results as
                :obj:`SegDataSample`. Defaults to False.
            batch_size (int): Batch size. Defaults to 1.
            show (bool): Whether to display the rendering color segmentation
                mask in a popup window. Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_dir (str): Output directory of inference results. Defaults
                to ''.
            img_out_dir (str): Subdirectory of `out_dir`, used to save
                rendering color segmentation mask, so `out_dir` must be defined
                if you would like to save predicted mask. Defaults to 'vis'.
            pred_out_dir (str): Subdirectory of `out_dir`, used to save
                predicted mask file, so `out_dir` must be defined if you would
                like to save predicted mask. Defaults to 'pred'.

            **kwargs: Other keyword arguments passed to :meth:`preprocess`,
                :meth:`forward`, :meth:`visualize` and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``, ``visualize_kwargs``
                and ``postprocess_kwargs``.


        Returns:
            dict: Inference and visualization results.
        """

        if out_dir != '':
            pred_out_dir = osp.join(out_dir, pred_out_dir)
            img_out_dir = osp.join(out_dir, img_out_dir)
        else:
            pred_out_dir = ''
            img_out_dir = ''

        return super().__call__(
            inputs=inputs,
            return_datasamples=return_datasamples,
            batch_size=batch_size,
            show=show,
            wait_time=wait_time,
            img_out_dir=img_out_dir,
            pred_out_dir=pred_out_dir,
            **kwargs)

    def visualize(self,
                  inputs: list,
                  preds: List[dict],
                  show: bool = False,
                  wait_time: int = 0,
                  img_out_dir: str = '',
                  opacity: float = 0.8) -> List[np.ndarray]:
        """Visualize predictions.

        Args:
            inputs (list): Inputs preprocessed by :meth:`_inputs_to_list`.
            preds (Any): Predictions of the model.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            img_out_dir (str): Output directory of rendering prediction i.e.
                color segmentation mask. Defaults: ''
            opacity (int, float): The transparency of segmentation mask.
                Defaults to 0.8.

        Returns:
            List[np.ndarray]: Visualization results.
        """
        if self.visualizer is None or (not show and img_out_dir == ''):
            return None

        if getattr(self, 'visualizer') is None:
            raise ValueError('Visualization needs the "visualizer" term'
                             'defined in the config, but got None')
        self.visualizer.set_dataset_meta(**self.model.dataset_meta)
        self.visualizer.alpha = opacity

        results = []

        for single_input, pred in zip(inputs, preds):
            if isinstance(single_input, str):
                img_bytes = mmengine.fileio.get(single_input)
                img = mmcv.imfrombytes(img_bytes)
                img = img[:, :, ::-1]
                img_name = osp.basename(single_input)
            elif isinstance(single_input, np.ndarray):
                img = single_input.copy()
                img_num = str(self.num_visualized_imgs).zfill(8) + '_vis'
                img_name = f'{img_num}.jpg'
            else:
                raise ValueError('Unsupported input type:'
                                 f'{type(single_input)}')

            out_file = osp.join(img_out_dir, img_name) if img_out_dir != ''\
                else None
            self.visualizer.add_datasample(
                img_name,
                img,
                pred,
                show=show,
                wait_time=wait_time,
                draw_gt=True,
                draw_pred=True,
                out_file=out_file)
            results.append(self.visualizer.get_image())
            self.num_visualized_imgs += 1

        return results

    def postprocess(self,
                    preds: PredType,
                    visualization: List[np.ndarray],
                    return_datasample: bool = False,
                    pred_out_dir: str = '') -> dict:
        """Process the predictions and visualization results from ``forward``
        and ``visualize``.

        This method should be responsible for the following tasks:

        1. Pack the predictions and visualization results and return them.
        2. Save the predictions, if it needed.

        Args:
            preds (List[Dict]): Predictions of the model.
            visualization (List[np.ndarray]): The list of rendering color
                segmentation mask.
            return_datasample (bool): Whether to return results as datasamples.
                Defaults to False.
            pred_out_dir: File to save the inference results w/o
                visualization. If left as empty, no file will be saved.
                Defaults to ''.

        Returns:
            dict: Inference and visualization results with key ``predictions``
            and ``visualization``

            - ``visualization (Any)``: Returned by :meth:`visualize`
            - ``predictions`` (List[np.ndarray], np.ndarray): Returned by
              :meth:`forward` and processed in :meth:`postprocess`.
              If ``return_datasample=False``, it will be the segmentation mask
              with label indice.
        """
        if return_datasample:
            if len(preds) == 1:
                return preds[0]
            else:
                return preds

        results_dict = {}

        results_dict['predictions'] = []
        results_dict['visualization'] = []

        for i, pred in enumerate(preds):
            pred_data = pred.pred_sem_seg.numpy().data[0]
            results_dict['predictions'].append(pred_data)
            if visualization is not None:
                vis = visualization[i]
                results_dict['visualization'].append(vis)
            if pred_out_dir != '':
                mmengine.mkdir_or_exist(pred_out_dir)
                img_name = str(self.num_pred_imgs).zfill(8) + '_pred.png'
                img_path = osp.join(pred_out_dir, img_name)
                output = Image.fromarray(pred_data.astype(np.uint8))
                output.save(img_path)
            self.num_pred_imgs += 1

        if len(results_dict['predictions']) == 1:
            results_dict['predictions'] = results_dict['predictions'][0]
            if visualization is not None:
                results_dict['visualization'] = \
                    results_dict['visualization'][0]
        return results_dict

    def _init_pipeline(self, cfg: ConfigType) -> Compose:
        """Initialize the test pipeline.

        Return a pipeline to handle various input data, such as ``str``,
        ``np.ndarray``. It is an abstract method in BaseInferencer, and should
        be implemented in subclasses.

        The returned pipeline will be used to process a single data.
        It will be used in :meth:`preprocess` like this:

        .. code-block:: python
            def preprocess(self, inputs, batch_size, **kwargs):
                ...
                dataset = map(self.pipeline, dataset)
                ...
        """
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline
        from mmpretrain.datasets import remove_transform
        # Loading annotations is also not applicable
        idx = self._get_transform_idx(pipeline_cfg, 'LoadAnnotations')
        if idx != -1:
            del pipeline_cfg[idx]
        load_img_idx = self._get_transform_idx(pipeline_cfg,
                                               'LoadImageFromFile')

        if load_img_idx == -1:
            raise ValueError(
                'LoadImageFromFile is not found in the test pipeline')
        pipeline_cfg[load_img_idx]['type'] = 'InferencerLoader'
        return Compose(pipeline_cfg)

    def _get_transform_idx(self, pipeline_cfg: ConfigType, name: str) -> int:
        """Returns the index of the transform in a pipeline.

        If the transform is not found, returns -1.
        """
        # for i, transform in enumerate(pipeline_cfg):
        #     if transform['type'] == name:
        #         return i
        # return -1
        for i, transform in enumerate(pipeline_cfg):
            if isinstance(transform, dict):
                if isinstance(transform['type'], type):
                    if transform['type'].__name__ == name:
                        return i
                else:
                    if transform['type'] == name:
                        return i
            else:
                if transform.__class__.__name__ == name:
                    return i

        return -1
