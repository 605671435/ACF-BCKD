# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Union
import os

import mmcv
import torch
import torch.nn.functional as F
import numpy as np
from mmengine.model import BaseModel
from mmengine.utils.path import mkdir_or_exist
from torch import nn
from mmseg.visualization import SegLocalVisualizer
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from mmpretrain.models.selfsup import BaseSelfSupervisor
from .ops import aug_rand, rot_rand
import matplotlib.pyplot as plt


def normalize(tensor: torch.Tensor):
    array = tensor.detach().cpu().numpy()
    array = (array - np.min(array)) / (np.max(array) - np.min(array))
    array = array[0][0] * 255.0
    array = array.astype(np.uint8)
    return array


class Contrast(torch.nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        device = torch.device(f"cuda:0")
        self.batch_size = batch_size
        self.register_buffer("temp", torch.tensor(temperature).to(torch.device(f"cuda:0")))
        self.register_buffer("neg_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())

    def forward(self, x_i, x_j):
        z_i = F.normalize(x_i, dim=1)
        z_j = F.normalize(x_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_ij = torch.diag(sim, self.batch_size)
        sim_ji = torch.diag(sim, -self.batch_size)
        pos = torch.cat([sim_ij, sim_ji], dim=0)
        nom = torch.exp(pos / self.temp)
        denom = self.neg_mask * torch.exp(sim / self.temp)
        return torch.sum(-torch.log(nom / torch.sum(denom, dim=1))) / (2 * self.batch_size)


class SwinUNETR_SelfSupervisor(BaseSelfSupervisor, metaclass=ABCMeta):
    """BaseModel for Self-Supervised Learning.

    All self-supervised algorithms should inherit this module.

    Args:
        backbone (dict): The backbone module. See
            :mod:`mmpretrain.models.backbones`.
        neck (dict, optional): The neck module to process features from
            backbone. See :mod:`mmpretrain.models.necks`. Defaults to None.
        head (dict, optional): The head module to do prediction and calculate
            loss from processed features. See :mod:`mmpretrain.models.heads`.
            Notice that if the head is not set, almost all methods cannot be
            used except :meth:`extract_feat`. Defaults to None.
        target_generator: (dict, optional): The target_generator module to
            generate targets for self-supervised learning optimization, such as
            HOG, extracted features from other modules(DALL-E, CLIP), etc.
        pretrained (str, optional): The pretrained checkpoint path, support
            local path and remote path. Defaults to None.
        data_preprocessor (Union[dict, nn.Module], optional): The config for
            preprocessing input data. If None or no specified type, it will use
            "SelfSupDataPreprocessor" as type.
            See :class:`SelfSupDataPreprocessor` for more details.
            Defaults to None.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """
    def __init__(self,
                 alpha1=1.0,
                 alpha2=1.0,
                 alpha3=1.0,
                 batch_size=2,
                 plot=True,
                 save_dir='./selfsup_outputs',
                 **kwargs):
        super().__init__(**kwargs)
        self.rot_loss = torch.nn.CrossEntropyLoss().cuda()
        self.recon_loss = torch.nn.L1Loss().cuda()
        self.contrast_loss = Contrast(batch_size).cuda()

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3

        self.plot = plot
        self.save_dir = save_dir
        if self.plot:
            mkdir_or_exist(self.save_dir)

    def forward(self,
                inputs: Union[torch.Tensor, List[torch.Tensor]],
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'tensor'):
        """The unified entry for a forward process in both training and test.

        The method currently accepts two modes: "tensor" and "loss":

        - "tensor": Forward the backbone network and return the feature
          tensor(s) tensor without any post-processing, same as a common
          PyTorch Module.
        - "loss": Forward and return a dict of losses according to the given
          inputs and data samples.

        Args:
            inputs (torch.Tensor or List[torch.Tensor]): The input tensor with
                shape (N, C, ...) in general.
            data_samples (List[DataSample], optional): The other data of
                every samples. It's required for some algorithms
                if ``mode="loss"``. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def loss(self, inputs: torch.Tensor,
             data_samples: List[DataSample]) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        This is a abstract method, and subclass should overwrite this methods
        if needed.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = inputs
        x1, rot1 = rot_rand(x)
        x2, rot2 = rot_rand(x)
        x1_augment = aug_rand(x1)
        x2_augment = aug_rand(x2)
        x1_augment = x1_augment
        x2_augment = x2_augment
        rot1_p, contrastive1_p, rec_x1 = self.backbone(x1_augment)
        rot2_p, contrastive2_p, rec_x2 = self.backbone(x2_augment)
        rot_p = torch.cat([rot1_p, rot2_p], dim=0)
        rots = torch.cat([rot1, rot2], dim=0)
        imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
        imgs = torch.cat([x1, x2], dim=0)

        losses = dict()
        loss = self.loss_function(
            inputs=(rot_p, rots, contrastive1_p, contrastive2_p, imgs_recon, imgs))
        losses.update(loss)

        return losses

    def predict(self, inputs: torch.Tensor,
                data_samples: List[DataSample]) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        This is a abstract method, and subclass should overwrite this methods
        if needed.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        val_inputs = inputs.cuda()
        x1, rot1 = rot_rand(val_inputs)
        x2, rot2 = rot_rand(val_inputs)
        x1_augment = aug_rand(x1)
        x2_augment = aug_rand(x2)
        rot1_p, contrastive1_p, rec_x1 = self.backbone(x1_augment) # noqa
        rot2_p, contrastive2_p, rec_x2 = self.backbone(x2_augment) # noqa
        rot_p = torch.cat([rot1_p, rot2_p], dim=0)
        rots = torch.cat([rot1, rot2], dim=0)
        imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
        imgs = torch.cat([x1, x2], dim=0)
        losses = dict()
        loss = self.loss_function(
            inputs=(rot_p, rots, contrastive1_p, contrastive2_p, imgs_recon, imgs))
        losses.update(loss)
        for k, v in losses.items():
            losses[k] = v.detach().cpu().numpy()

        # x_gt = x1.detach().cpu().numpy()
        # x_gt = (x_gt - np.min(x_gt)) / (np.max(x_gt) - np.min(x_gt))
        # xgt = x_gt[0][0] * 255.0
        # xgt = xgt.astype(np.uint8)
        #
        # x1_augment = x1_augment.detach().cpu().numpy()
        # x1_augment = (x1_augment - np.min(x1_augment)) / (np.max(x1_augment) - np.min(x1_augment))
        # x_aug = x1_augment[0][0] * 255.0
        # x_aug = x_aug.astype(np.uint8)
        #
        # rec_x1 = rec_x1.detach().cpu().numpy()
        # rec_x1 = (rec_x1 - np.min(rec_x1)) / (np.max(rec_x1) - np.min(rec_x1))
        # recon = rec_x1[0][0] * 255.0
        # recon = recon.astype(np.uint8)

        if self.plot:
            self.plot_pred(
                images=[x1, x1_augment, rec_x1],
                # images=[xgt, x_aug, recon],
                filename=os.path.basename(data_samples[0].img_path)) # noqa
        return losses

    def loss_function(self, inputs):
        loss = dict()

        output_rot, target_rot, output_contrastive, target_contrastive, output_recons, target_recons = inputs

        loss['rot_loss'] = self.alpha1 * self.rot_loss(output_rot, target_rot.long())
        loss['contrast_loss'] = self.alpha2 * self.contrast_loss(output_contrastive, target_contrastive)
        loss['recon_loss'] = self.alpha3 * self.recon_loss(output_recons, target_recons)

        return loss

    def plot_pred(self, images: List[torch.Tensor], filename: str):
        # 创建一个1x3的子图布局
        plt.figure(figsize=(12, 4))  # 设置整体图的大小
        titles = ['ground truth', 'augmented image', 'reconstructed image']
        for i, image in enumerate(images):
            image = normalize(image)
            plt.subplot(1, 3, i)
            plt.imshow(image)  # 使用imshow显示图像
            plt.title(titles[i])  # 设置标题
        # # 第一个子图
        # plt.subplot(1, 3, 1)
        # plt.imshow(images[0])  # 使用imshow显示图像
        # plt.title('ground truth')  # 设置标题
        # # 第二个子图
        # plt.subplot(1, 3, 2)
        # plt.imshow(images[1])
        # plt.title('augmented image')
        # # 第三个子图
        # plt.subplot(1, 3, 3)
        # plt.imshow(images[2])
        # plt.title('reconstructed image')
        # # plt.show()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()
