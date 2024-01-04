from dynamic_network_architectures.architectures.unet import PlainConvUNet
from typing import Union, Type, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from mmseg.utils import ConfigType, SampleList
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize
from .decode_head import BaseDecodeHead

class UNet(BaseDecodeHead):

    def __init__(self,
                 loss_decode,
                 in_channels=1,
                 channels=3,
                 resize_mode='bilinear',
                 align_corners=False,
                 n_stages: int = 6,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]] = [32, 64, 128, 256, 512, 512],
                 conv_op: Type[_ConvNd] = nn.Conv2d,
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]] = [[3,3],[3,3],[3,3],[3,3],[3,3],[3,3]],
                 strides: Union[int, List[int], Tuple[int, ...]] = [[1,1],[2,2],[2,2],[2,2],[2,2],[2,2]],
                 num_classes: int = 9,
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]] = [2, 2, 2, 2, 2, 2],
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]] = [2, 2, 2, 2, 2],
                 conv_bias: bool = True,
                 norm_op: Union[None, Type[nn.Module]] = nn.InstanceNorm2d,
                 norm_op_kwargs: dict = {'eps': 1e-5, 'affine': True},
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = nn.LeakyReLU,
                 nonlin_kwargs: dict = {'inplace': True},
                 deep_supervision: bool = True,
                 nonlin_first: bool = False,
                 weight_factors=None,
                 **kwargs):
        super().__init__(
            num_classes=num_classes,
            loss_decode=loss_decode,
            in_channels=in_channels,
            channels=channels,
            resize_mode=resize_mode,
            align_corners=align_corners,
            **kwargs)
        self.weight_factors = self._get_deep_supervision_weights(strides)
        self.unet = PlainConvUNet(
            in_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_conv_per_stage,
            self.num_classes,
            n_conv_per_stage_decoder,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            deep_supervision,
            nonlin_first
        )

    def _get_deep_supervision_weights(self, pool_op_kernel_sizes):
        deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
            pool_op_kernel_sizes), axis=0))[:-1]
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0
        weights = weights / weights.sum()
        return weights

    def forward(self, inputs):
        """Forward function."""
        output = self.unet(inputs)
        return output

    def loss_by_feat(self, seg_logits: List,
                     batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_label = self._stack_batch_gt(batch_data_samples)
        seg_labels = []
        for i in range(len(seg_logits)):
            seg_labels.append(
                resize(
                    input=seg_label.float(),
                    size=seg_logits[i].shape[-2:],
                    mode='bilinear',
                    align_corners=self.align_corners).type(seg_label.dtype))

        loss = dict()

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_labels = [label.squeeze(1) for label in seg_labels]

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            l = self.weight_factors[0] * loss_decode(
                seg_logits[0],
                seg_labels[0],
                weight=seg_weight,
                ignore_index=self.ignore_index)
            for i, (logits, labels) in enumerate(zip(seg_logits, seg_labels)):
                if i == 0:
                    continue
                l += self.weight_factors[i] * loss_decode(
                    logits,
                    labels,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = l
            else:
                loss[loss_decode.loss_name] += l

        loss['acc_seg'] = accuracy(
            seg_logits[0], seg_labels[0], ignore_index=self.ignore_index)
        return loss

    def predict_by_feat(self, seg_logits: List,
                        batch_img_metas: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        if len(batch_img_metas[0]['img_shape']) > 2:
            size = (batch_img_metas[0]['img_shape'][2],) + batch_img_metas[0]['img_shape'][:2]
        else:
            size = batch_img_metas[0]['img_shape']
        seg_logits = resize(
            input=seg_logits[0],
            size=size,
            mode=self.resize_mode,
            align_corners=self.align_corners)
        return seg_logits
