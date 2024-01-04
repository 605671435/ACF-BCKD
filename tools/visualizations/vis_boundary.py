import os
import os.path as osp

import torch
import torch.nn.functional as F
import numpy as np

import mmcv
import nibabel as nib
import argparse
import matplotlib.pyplot as plt

from monai import transforms
from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner import load_checkpoint
from mmengine.utils.path import mkdir_or_exist

from utils.utils import resample_3d, get_timestamp

if __name__ == '__main__':
    img_path = 'data/synapse_raw/imagesTr/img0038.nii.gz'
    label_path = img_path.replace('imagesTr', 'labelsTr').replace('img', 'label')
    img = nib.load(img_path).get_fdata()
    label = nib.load(label_path).get_fdata()

    gt_cls = label[..., 80] == 7
    gt_cls = torch.tensor(gt_cls).unsqueeze(0)
    kernel = torch.ones((3, 3), dtype=torch.float32, device='cpu')
    kernel[1:-1, 1:-1] = 0
    kernel = kernel.view(1, 1, 3, 3)
    boundary = F.conv2d(gt_cls.float(), kernel, padding=1)
    boundary[boundary == kernel.sum()] = 0
    boundary[boundary > 0] = 1
    plt.imshow(boundary[0])
    plt.show()

    boundary_rgb = torch.cat([boundary, boundary, boundary], dim=0)
    boundary_rgb = boundary_rgb.permute(1, 2, 0).numpy()
    boundary_rgb[..., 0] = boundary_rgb[..., 0] * 213
    boundary_rgb[..., 1] = boundary_rgb[..., 1] * 239
    boundary_rgb[..., 2] = boundary_rgb[..., 2] * 255
    mmcv.imwrite(img=boundary_rgb, file_path='save_dirs/paper_vis/boundary_7.jpg')

    teacher_label = nib.load('save_dirs/paper_vis/pred0038_teacher.nii.gz').get_fdata()
    student_label = nib.load('save_dirs/paper_vis/pred0038_student.nii.gz').get_fdata()
    plt.imshow(student_label[..., 80])
    plt.show()
    plt.imshow(teacher_label[..., 80])
    plt.show()
    plt.imshow(label[..., 80])
    plt.show()
    print()