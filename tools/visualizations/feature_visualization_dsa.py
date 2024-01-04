# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import mmcv
import torch
from mmengine.config import Config, DictAction
from mmengine.registry import VISUALIZERS

from seg.recorder import RecorderManager
from mmrazor.models.task_modules import ModuleOutputsRecorder, ModuleInputsRecorder
from mmrazor.visualization.local_visualizer import modify
# from seg.utils.misc import find_best_in_multi_run
from seg.apis import init_model, inference_model
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Feature map visualization')
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='train config file path')
    # parser.add_argument('vis_config', help='visualization config file path')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda', help='Device used for inference')
    parser.add_argument('--repo', help='the corresponding repo name')
    parser.add_argument(
        '--use-norm',
        action='store_true',
        help='normalize the featmap before visualization')
    parser.add_argument(
        '--overlaid', action='store_true', help='overlaid image')
    parser.add_argument(
        '--channel-reduction',
        help='Reduce multiple channels to a single channel. The optional value'
             ' is \'squeeze_mean\', \'select_max\' or \'pixel_wise_max\'.',
        default=None)
    parser.add_argument(
        '--topk',
        type=int,
        help='If channel_reduction is not None and topk > 0, it will select '
             'topk channel to show by the sum of each channel. If topk <= 0, '
             'tensor_chw is assert to be one or three.',
        default=20)
    parser.add_argument(
        '--arrangement',
        nargs='+',
        type=int,
        help='the arrangement of featmap when channel_reduction is not None '
             'and topk > 0.',
        default=[4, 5])
    parser.add_argument(
        '--resize-shape',
        nargs='+',
        type=int,
        help='the shape to scale the feature map',
        default=None)
    parser.add_argument(
        '--alpha', help='the transparency of featmap', default=0.5)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.',
        default={})

    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def norm(feat):
    N, C, H, W = feat.shape
    feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
    mean = feat.mean(dim=-1, keepdim=True)
    std = feat.std(dim=-1, keepdim=True)
    centered = (feat - mean) / (std + 1e-6)
    centered = centered.reshape(C, N, H, W).permute(1, 0, 2, 3)
    return centered

config_mapping = [
    # dict(
    #     config_path='new_configs/se/unet_r18v1c_se_40k_synapse.py',
    #     checkpoint='vis_ckpts/unet_r18v1c_se_40k_synapse/best_mDice_75-51_iter_40000.pth',
    #     attn_module='se_layer'),
    # dict(
    #     config_path='new_configs/cbam/unet_r18v1c_cbam_40k_synapse.py',
    #     checkpoint='vis_ckpts/unet_r18v1c_cbam_40k_synapse/best_mDice_75-72_iter_36000.pth',
    #     # attn_module='cbam'),
    #     attn_module='SpatialGate'),
    # dict(
    #     config_path='new_configs/psa/unet_r18v1c_psa_s_40k_synapse.py',
    #     checkpoint='vis_ckpts/unet_r18v1c_psa_s_40k_synapse/best_mDice_75-76_iter_40000.pth',
    #     attn_module='psa_s'),
    # dict(
    #     config_path='new_configs/ccnet/unet_r18v1c_ccnet_40k_synapse.py',
    #     checkpoint='vis_ckpts/unet_r18v1c_ccnet_40k_synapse/best_mDice_76-59_iter_40000.pth',
    #     attn_module='criss_cross_attention'),
    # dict(
    #     config_path='new_configs/gcnet/unet_r18v1c_gcb_40k_synapse.py',
    #     checkpoint='vis_ckpts/unet_r18v1c_gcb_40k_synapse/best_mDice_76-16_iter_40000.pth',
    #     attn_module='context_block'),
    # dict(
    #     config_path='new_configs/hamnet/unet_r18v1c_hamnet_40k_synapse.py',
    #     checkpoint='vis_ckpts/unet_r18v1c_hamnet_40k_synapse/best_mDice_77-43_iter_40000.pth',
    #     attn_module='ham'),
    # dict(
    #     config_path='new_configs/attn_unet/attn_ma_unet_r18v1c_synapse_40k.py',
    #     checkpoint='work_dirs/attn_ma_unet_r18v1c_synapse_40k/5-run_20231013_165701/run0/best_mDice_77-77_iter_36000.pth',
    #     attn_module='MultiAttentionBlock2D'),
    # dict(
    #     config_path='new_configs/dsnet/unet_r18v1c_dsnet_v14_40k_synapse.py',
    #     checkpoint='vis_ckpts/unet_r18v1c_dsnet_v14_40k_synapse/best_mDice_79-30_iter_36000.pth',
    #     # attn_module='dsa_v14'),
    #     attn_module='sp_attn'),
    dict(
        config_path='new_configs/dsnet/unet_r18v1c_dsnet_v14_dam_40k_synapse.py',
        checkpoint='vis_ckpts/unet_r18v1c_dsnet_v14_dam_40k_synapse/best_mDice_77-88_iter_40000.pth',
        # attn_module='dsa_v14'),
        attn_module='sp_attn'),
]

def main(args, method):

    # config = config_mapping[method]['config_path']
    # checkpoint = config_mapping[method]['work_dir']
    # attn_module = config_mapping[method]['attn_module']

    model = init_model(args.config, args.checkpoint, device=args.device)

    recorders = dict()
    for name, module in model.named_modules():
        # if name.endswith(attn_module):
        if name.endswith(method):
            recorders[name.replace('.', '_')] = dict(type=ModuleOutputsRecorder, source=name)
    assert len(recorders) > 0

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.draw_featmap = modify

    recorder_manager = RecorderManager(recorders)
    recorder_manager.initialize(model)

    with recorder_manager:
        # test a single image
        result = inference_model(model, args.img, args.img.replace('img_dir', 'ann_dir').replace('jpg', 'png'))

    # visualizer.add_datasample(
    #                 osp.splitext(osp.basename(img_path))[0],
    #                 img,
    #                 result,
    #                 draw_gt=True,
    #                 draw_pred=False,
    #                 out_file=osp.join('./out_dir/gt', dataset_name, osp.basename(img_path)))

    overlaid_image = mmcv.imread(
        args.img, channel_order='rgb') if args.overlaid else None

    for name in recorders.keys():
        recorder = recorder_manager.get_recorder(name)
        # record_idx = getattr(name, 'record_idx', 0)
        # data_idx = getattr(name, 'data_idx')
        feats = recorder.get_record_data()
        if isinstance(feats, torch.Tensor):
            feats = (feats,)

        for i, feat in enumerate(feats):
            if args.use_norm:
                feat = norm(feat)
            drawn_img = visualizer.draw_featmap(
                feat[0],
                overlaid_image,
                args.channel_reduction,
                topk=args.topk,
                arrangement=tuple(args.arrangement),
                resize_shape=tuple(args.resize_shape)
                if args.resize_shape else None,
                alpha=args.alpha)
            mmcv.imwrite(mmcv.rgb2bgr(drawn_img),
                         f'./feat_maps/{osp.splitext(osp.basename(args.config))[0]}/{name}.jpg')
            # plt.imshow(drawn_img)
            # plt.show()
            # visualizer.add_datasample(
            #     f'{name}_{i}',
            #     drawn_img,
            #     data_sample=result,
            #     draw_gt=False,
            #     show=args.out_file is None,
            #     wait_time=0.1,
            #     out_file=args.out_file,
            #     **args.cfg_options)


if __name__ == '__main__':
    args = parse_args()
    main(args, 'combine_attn')
    # for k in config_mapping:
    #     args.config = k['config_path']
    #     args.checkpoint = k['checkpoint']
    #     main(args, 'backbone.layer4')
