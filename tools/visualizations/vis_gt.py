import os
import os.path as osp

import torch
import numpy as np

import nibabel as nib
import argparse
import matplotlib.pyplot as plt

from monai import transforms
from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner import load_checkpoint
from mmengine.utils.path import mkdir_or_exist

from utils.utils import resample_3d, get_timestamp


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '-img-path',
        default='data/synapse_raw/imagesTr/img0038.nii.gz',
        help='train config file path')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--save', action='store_true', help='save prediction results')
    parser.add_argument(
        '--cuda', action='store_true', help='save prediction results')
    args = parser.parse_args()
    return args


def vis_gt(args):

    data_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], image_only=False),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            # transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys="image", pixdim=(1.5, 1.5, 2.0), mode="bilinear"
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175.0, a_max=250.0, b_min=0.0, b_max=1.0, clip=True
            ),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    data = data_transform(
        {'image': args.img_path,
         'label': args.img_path.replace('imagesTr', 'labelsTr').replace('img', 'label')})
    if args.show:
        plt.imshow(data['image'].squeeze()[..., -1 - 66])
        plt.show()
        plt.imshow(data['label'].squeeze()[..., -1 - 66])
        plt.show()
    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint)
    data['image'] = data['image'].unsqueeze(1)
    data['label'] = data['label'].unsqueeze(1)
    if args.cuda:
        model = model.cuda()
        data['image'] = data['image'].cuda()
        # data['label'] = data['label'].cuda()
    outputs = model.test_step(data)
    outputs = torch.softmax(outputs, 1)
    outputs = torch.argmax(outputs, 1)
    if args.show:
        plt.imshow(outputs.squeeze()[..., -1 - 66])
        plt.show()
    if args.save:
        # nii_img = nib.load(args.img_path)

        # shape = nii_img.get_fdata().shape
        # affine = nii_img.affine
        shape = data['label_meta_dict']['spatial_shape'][0]
        affine = data['label_meta_dict']['original_affine'][0]
        outputs = outputs.cpu().numpy().astype(np.uint8)[0]
        val_outputs = resample_3d(outputs, shape)

        target_dir = os.path.join('./preds', osp.basename(args.config))

        mkdir_or_exist(target_dir)

        target_file = os.path.join(
                target_dir,
                get_timestamp() + '_' + osp.basename(
                    args.img_path.replace('imagesTr', 'labelsTr').replace('img', 'label')))

        nib.save(
            nib.Nifti1Image(val_outputs.astype(np.uint8), affine),
            target_file
        )
        print(f'image saves to {target_file}')

def main():
    args = parse_args()
    # img_dir = '/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val'
    # img_dir = '/home/jz207/workspace/zhangdw/ex_kd/data/FLARE22/img_dir/val'
    # pb1 = ProgressBar(len(os.listdir(img_dir)))
    # for case in os.listdir(img_dir):
    #     pb2 = ProgressBar(len(os.listdir(osp.join(img_dir, case))))
    #     for img in os.listdir(osp.join(img_dir, case)):
    #         vis_gt(osp.join(img_dir, case, img), dataset_name='flare22')
    #         pb2.update()
    #     pb1.update()
    vis_gt(args)

if __name__=='__main__':
    main()
    # vis_gt('/home/jz207/workspace/zhangdw/ex_kd/data/synapse_raw/imagesTr/img0022.nii.gz')

teacher='configs/unet/unetmod_base_d8_1000e_sgd_synapse_96x96x96.py ckpts/unetmod_base_d8_1000e_sgd_synapse_96x96x96/best_Dice_81-69_epoch_800.pth --save'
kd='configs/unet/unetmod_tiny_d8_1000e_sgd_synapse_96x96x96.py work_dirs/boundarykd_unet_base_d8_unet_tiny_d8_1000e_sgd_synapse_96x96x96/3-run_20231114_190801/run2/best_Dice_76-76_epoch_1000_student.pth /home/jz207/workspace/zhangdw/ex_kd/data/synapse_raw/imagesTr/img0022.nii.gz'
kd2='work_dirs/eta050_loghdkd_bkdv1_exkd_v16_unet_base_d8_unet_tiny_d8_1000e_sgd_synapse_96x96x96/3-run_20231118_000548/run1/best_Dice_76-64_epoch_1000_student.pth'
student='work_dirs/unetmod_tiny_d8_1000e_sgd_synapse_96x96x96/3-run_20231107_001947/run0/best_Dice_66-03_epoch_1000_student.pth'
swin='configs/swin_unetr/swinunetr_base_5000e_synapse.py ckpts/swin_unetr.base_5000ep_f48_lr2e-4_pretrained_mmengine.pth'