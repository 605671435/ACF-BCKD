import os
from typing import Dict, List, Union
import numpy as np

from mmengine.config import Config, ConfigDict
from mmengine.optim import (OptimWrapper, OptimWrapperDict, _ParamScheduler,)

from monai import data, transforms

from seg.datasets.sampler import Sampler
from seg.datasets.monai_dataset import MonaiDataset, CacheMonaiDataset, SmartCacheMonaiDataset

ConfigType = Union[Dict, Config, ConfigDict]
ParamSchedulerType = Union[List[_ParamScheduler], Dict[str,
                                                       List[_ParamScheduler]]]
OptimWrapperType = Union[OptimWrapper, OptimWrapperDict]


def BTCV_loader(args: ConfigType, test_mode: bool, save: bool):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], image_only=False),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], image_only=False),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            # transforms.Orientationd(keys=["image"], axcodes="RAS"),
            transforms.Spacingd(keys="image", pixdim=(args.space_x, args.space_y, args.space_z), mode="bilinear"),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if test_mode:
        if save:
            test_files = data.load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
            test_ds = MonaiDataset(meta_info=args.meta_info, data=test_files, transform=test_transform)
            test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
            test_loader = data.DataLoader(
                test_ds,
                batch_size=1,
                shuffle=False,
                num_workers=args.workers,
                sampler=test_sampler,
                pin_memory=True,
                persistent_workers=True,
            )
            loader = test_loader
        else:
            val_files = data.load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
            val_ds = MonaiDataset(meta_info=args.meta_info, data=val_files, transform=val_transform)
            val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
            val_loader = data.DataLoader(
                val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
            )
            loader = val_loader
    else:
        datalist = data.load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
        if args.use_normal_dataset:
            train_ds = MonaiDataset(meta_info=args.meta_info, data=datalist, transform=train_transform)
        else:
            train_ds = CacheMonaiDataset(
                meta_info=args.meta_info,
                data=datalist,
                transform=train_transform,
                cache_num=args.train_case_nums,
                cache_rate=1.0,
                num_workers=args.workers
            )
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
        val_files = data.load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        val_ds = MonaiDataset(meta_info=args.meta_info, data=val_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )
        loader = [train_loader, val_loader]

    return loader


def WORD_loader(args: ConfigType, test_mode: bool, save: bool):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], image_only=False),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], image_only=False),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            # transforms.Orientationd(keys=["image"], axcodes="RAS"),
            transforms.Spacingd(keys="image", pixdim=(args.space_x, args.space_y, args.space_z), mode="bilinear"),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if test_mode:
        if save:
            test_files = data.load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
            test_ds = MonaiDataset(meta_info=args.meta_info, data=test_files, transform=test_transform)
            test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
            test_loader = data.DataLoader(
                test_ds,
                batch_size=1,
                shuffle=False,
                num_workers=args.workers,
                sampler=test_sampler,
                pin_memory=True,
                persistent_workers=True,
            )
            loader = test_loader
        else:
            val_files = data.load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
            val_ds = MonaiDataset(meta_info=args.meta_info, data=val_files, transform=val_transform)
            val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
            val_loader = data.DataLoader(
                val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
            )
            loader = val_loader
    else:
        datalist = data.load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
        if args.use_normal_dataset:
            train_ds = MonaiDataset(meta_info=args.meta_info, data=datalist, transform=train_transform)
        else:
            train_ds = CacheMonaiDataset(
                meta_info=args.meta_info,
                data=datalist,
                transform=train_transform,
                cache_num=args.train_case_nums,
                cache_rate=1.0,
                num_workers=args.workers
            )
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
        val_files = data.load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        val_ds = MonaiDataset(meta_info=args.meta_info, data=val_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )
        loader = [train_loader, val_loader]

    return loader

# def WORD_loader(args: ConfigType, test_mode: bool, save: bool):
#     data_dir = args.data_dir
#     datalist_json = os.path.join(data_dir, args.json_list)
#     train_transform = transforms.Compose(
#         [
#             transforms.LoadImaged(keys=["image", "label"]),
#             transforms.EnsureChannelFirstd(keys=["image", "label"]),
#             transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
#             transforms.Spacingd(
#                 keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
#             ),
#             transforms.CropForegroundd(
#                 keys=["image", "label"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]
#             ),
#             transforms.RandSpatialCropd(
#                 keys=["image", "label"], roi_size=[args.roi_x, args.roi_y, args.roi_z], random_size=False
#             ),
#             transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
#             transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
#             transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
#             transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
#             transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
#             transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
#             transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
#             transforms.ToTensord(keys=["image", "label"]),
#         ]
#     )
#     val_transform = transforms.Compose(
#         [
#             transforms.LoadImaged(keys=["image", "label"], image_only=False),
#             transforms.EnsureChannelFirstd(keys=["image", "label"]),
#             transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
#             transforms.Spacingd(
#                 keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
#             ),
#             transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
#             transforms.ToTensord(keys=["image", "label"]),
#         ]
#     )
#
#     test_transform = transforms.Compose(
#         [
#             transforms.LoadImaged(keys=["image", "label"], image_only=False),
#             transforms.EnsureChannelFirstd(keys=["image", "label"]),
#             # transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
#             transforms.Spacingd(
#                 keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
#             ),
#             transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
#             transforms.ToTensord(keys=["image", "label"]),
#         ]
#     )
#
#     if test_mode:
#         if save:
#             test_files = data.load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
#             test_ds = MonaiDataset(meta_info=args.meta_info, data=test_files, transform=test_transform)
#             test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
#             test_loader = data.DataLoader(
#                 test_ds,
#                 batch_size=1,
#                 shuffle=False,
#                 num_workers=args.workers,
#                 sampler=test_sampler,
#                 pin_memory=True,
#                 persistent_workers=True,
#             )
#             loader = test_loader
#         else:
#             val_files = data.load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
#             val_ds = MonaiDataset(meta_info=args.meta_info, data=val_files, transform=val_transform)
#             val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
#             val_loader = data.DataLoader(
#                 val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
#             )
#             loader = val_loader
#     else:
#         datalist = data.load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
#         if args.use_normal_dataset:
#             train_ds = MonaiDataset(meta_info=args.meta_info, data=datalist, transform=train_transform)
#         elif args.use_smart_dataset:
#             train_ds = SmartCacheMonaiDataset(
#                 meta_info=args.meta_info,
#                 data=datalist,
#                 transform=train_transform,
#                 # cache_num=args.train_case_nums,
#                 cache_rate=0.25,
#                 num_init_workers=args.workers // 2,
#                 num_replace_workers=args.workers // 2
#             )
#         else:
#             train_ds = CacheMonaiDataset(
#                 meta_info=args.meta_info,
#                 data=datalist,
#                 transform=train_transform,
#                 cache_num=args.train_case_nums,
#                 cache_rate=1.0,
#                 num_workers=args.workers
#             )
#         train_sampler = Sampler(train_ds) if args.distributed else None
#         train_loader = data.DataLoader(
#             train_ds,
#             batch_size=args.batch_size,
#             shuffle=(train_sampler is None),
#             num_workers=args.workers,
#             sampler=train_sampler,
#             pin_memory=True,
#         )
#         val_files = data.load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
#         val_ds = MonaiDataset(meta_info=args.meta_info, data=val_files, transform=val_transform)
#         val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
#         val_loader = data.DataLoader(
#             val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
#         )
#         loader = [train_loader, val_loader]
#
#     return loader

# https://github.com/deepdrivepl/kits23/blob/main/params/unet.py
def KiTS23_loader(args: ConfigType, test_mode: bool, save: bool):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(
                keys=["image", "label"],
                image_only=True,
                ensure_channel_first=True,
            ),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ),
            transforms.NormalizeIntensityd(keys="image", nonzero=True),
            # ScaleIntensityd(keys="image"),
            transforms.SpatialPadd(
                keys=("image", "label"),
                spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
            transforms.RandCropByPosNegLabeld(
                keys=("image", "label"),
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
                allow_smaller=True,
            ),
            transforms.RandAffined(
                keys=("image", "label"),
                prob=0.75,
                rotate_range=(np.pi / 4, np.pi / 4),
                translate_range=(0.0625, 0.0625),
                scale_range=(0.1, 0.1),
            ),
            transforms.RandFlipd(
                keys=("image", "label"), spatial_axis=0, prob=0.5
            ),
            transforms.RandFlipd(
                keys=("image", "label"), spatial_axis=1, prob=0.5
            ),
            transforms.RandGaussianNoised(keys="image", prob=0.15, mean=0.0, std=0.01),
            transforms.RandGaussianSmoothd(
                keys="image", prob=0.15, sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15)
            ),
            transforms.RandScaleIntensityd(keys="image", factors=0.3, prob=0.15),
            # RandZoomd(
            #     keys=("image", "label"),
            #     min_zoom=0.9,
            #     max_zoom=1.2,
            #     mode=("bilinear", "nearest"),
            #     align_corners=(True, None),
            #     prob=0.15,
            # ),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(
                keys=["image", "label"],
                image_only=True,
                ensure_channel_first=True,
            ),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            transforms.NormalizeIntensityd(keys="image", nonzero=True),
            # ScaleIntensityd(keys="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = val_transform
    # test_transform = transforms.Compose(
    #     [
    #         transforms.LoadImaged(keys=["image", "label"], image_only=False),
    #         transforms.EnsureChannelFirstd(keys=["image", "label"]),
    #         # transforms.Orientationd(keys=["image"], axcodes="RAS"),
    #         transforms.Spacingd(keys="image", pixdim=(args.space_x, args.space_y, args.space_z), mode="bilinear"),
    #         transforms.ScaleIntensityRanged(
    #             keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
    #         ),
    #         transforms.ToTensord(keys=["image", "label"]),
    #     ]
    # )

    if test_mode:
        if save:
            test_files = data.load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
            test_ds = MonaiDataset(meta_info=args.meta_info, data=test_files, transform=test_transform)
            test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
            test_loader = data.DataLoader(
                test_ds,
                batch_size=1,
                shuffle=False,
                num_workers=args.workers,
                sampler=test_sampler,
                pin_memory=True,
                persistent_workers=True,
            )
            loader = test_loader
        else:
            val_files = data.load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
            val_ds = MonaiDataset(meta_info=args.meta_info, data=val_files, transform=val_transform)
            val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
            val_loader = data.DataLoader(
                val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
            )
            loader = val_loader
    else:
        datalist = data.load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
        datalist = datalist[:100]
        if args.use_normal_dataset:
            train_ds = MonaiDataset(meta_info=args.meta_info, data=datalist, transform=train_transform)
        else:
            train_ds = CacheMonaiDataset(
                meta_info=args.meta_info,
                data=datalist,
                transform=train_transform,
                cache_num=args.train_case_nums,
                cache_rate=1.0,
                num_workers=args.workers
            )
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
        val_files = data.load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        val_ds = MonaiDataset(meta_info=args.meta_info, data=val_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )
        loader = [train_loader, val_loader]

    return loader


def brats21_loader(args, test_mode: bool, save: bool):
    from seg.datasets import transforms as my_transforms
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            my_transforms.ConvertToMultiChannelBasedOnBrats23Classesd(keys="label"),
            transforms.CropForegroundd(
                keys=["image", "label"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]
            ),
            transforms.RandSpatialCropd(
                keys=["image", "label"], roi_size=[args.roi_x, args.roi_y, args.roi_z], random_size=False
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], image_only=False),
            my_transforms.ConvertToMultiChannelBasedOnBrats23Classesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            my_transforms.ConvertToMultiChannelBasedOnBrats23Classesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if test_mode:
        validation_files = data.load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        val_ds = MonaiDataset(meta_info=args.meta_info, data=validation_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )

        loader = test_loader
    else:
        train_files = data.load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
        if args.use_normal_dataset:
            train_ds = MonaiDataset(meta_info=args.meta_info, data=train_files, transform=train_transform)
        else:
            train_ds = CacheMonaiDataset(
                meta_info=args.meta_info,
                data=train_files,
                transform=train_transform,
                cache_num=args.train_case_nums,
                cache_rate=1.0,
                num_workers=args.workers
            )
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
        validation_files = data.load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        val_ds = MonaiDataset(meta_info=args.meta_info, data=validation_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )
        loader = [train_loader, val_loader]

    return loader


def brats23_loader(args, test_mode: bool, save: bool):
    from seg.datasets import transforms as my_transforms
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            # my_transforms.ConvertToMultiChannelBasedOnBrats23Classesd(keys="label"),
            transforms.CropForegroundd(
                keys=["image", "label"], source_key="image"
            ),
            transforms.RandSpatialCropd(
                keys=["image", "label"], roi_size=[args.roi_x, args.roi_y, args.roi_z], random_size=False
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], image_only=False, ensure_channel_first=True),
            # my_transforms.ConvertToMultiChannelBasedOnBrats23Classesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            # my_transforms.ConvertToMultiChannelBasedOnBrats23Classesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if test_mode:
        validation_files = data.load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        val_ds = MonaiDataset(meta_info=args.meta_info, data=validation_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )

        loader = test_loader
    else:
        train_files = data.load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
        if args.use_normal_dataset:
            train_ds = MonaiDataset(meta_info=args.meta_info, data=train_files, transform=train_transform)
        else:
            train_ds = CacheMonaiDataset(
                meta_info=args.meta_info,
                data=train_files,
                transform=train_transform,
                cache_num=args.train_case_nums,
                cache_rate=1.0,
                num_workers=args.workers
            )
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
        validation_files = data.load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        val_ds = MonaiDataset(meta_info=args.meta_info, data=validation_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )
        loader = [train_loader, val_loader]

    return loader
