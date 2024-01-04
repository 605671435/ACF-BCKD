from seg.apis import init_model, inference_model, show_result_pyplot
from seg.utils import register_all_modules
import medpy.metric as metric
import numpy as np
import os
import os.path as osp
from mmengine.fileio import dump
from mmengine.utils import ProgressBar
def compute_dice(pred, gt, num_classes, ignore_index):
    dice = []
    def calculate_metric_percase(pred, gt):
        pred[pred > 0] = 1
        gt[gt > 0] = 1
        if pred.sum() > 0 and gt.sum() > 0:
            dice = metric.binary.dc(pred, gt)
            return dice
        elif pred.sum() > 0 and gt.sum() == 0:
            return 1
        else:
            return 0
    for c in range(num_classes):
        if c == ignore_index:
            continue
        else:
            dice.append(calculate_metric_percase(pred == c, gt == c))
    mdice = np.mean(dice)
    return np.round(mdice*100, 2)

register_all_modules(init_default_scope=False)

# config_path = './configs/dsa/fcn_r50-ex-d8_1xb2-40k_synapse-512x512.py'
# checkpoint_path = './ckpt/work_dirs/fcn_r50-ex-d8_1xb8-40k_synapse-512x512/5-run_20230426_145858/run1/best_mDice_84-88_iter_32000.pth'

config_path = './configs/dsa/fcn_r50-ex-d8_1xb2-40k_acdc-256x256.py'
checkpoint_path = './ckpt/work_dirs/fcn_r50-ex-d8_1xb8-40k_acdc-512x512/5-run_20230430_125324/run1/best_mDice_91-16_iter_40000.pth'
# img_path = './data/synapse9/img_dir/val/case0036/case0036_slice130.jpg'
# seg_path = './data/synapse9/ann_dir/val/case0036/case0036_slice130.png'
config_name = osp.splitext(osp.basename(config_path))[0]
def scan_list(file_dir):
    file_list = []
    dir_list = os.listdir(file_dir)
    for dir in dir_list:
        full_dir = osp.join(file_dir, dir)
        sub_dir_list = os.listdir(full_dir)
        sub_dir_list = [osp.join(full_dir, sub_dir) for sub_dir in sub_dir_list]
        file_list += sub_dir_list
    return file_list

from ..configs._base_.datasets.acdc import ACDC_val_list

img_list = scan_list('./data/acdc/img_dir/val')
ann_list = scan_list('./data/acdc/ann_dir/val')
img_list.sort()
ann_list.sort()
# 从配置文件和权重文件构建模型
model = init_model(config_path, checkpoint_path, device='cuda:0')

def infer_on_image(img_path, seg_path):
    # 推理给定图像
    result = inference_model(model, img_path, seg_path)
    img_name = osp.splitext(osp.basename(img_path))[0]
    case_name = img_name.split('_')[0]
    gt_array = result.gt_sem_seg.data.squeeze(0).cpu().numpy()
    pred_array = result.pred_sem_seg.data.squeeze(0).cpu().numpy()

    dice = compute_dice(pred_array, gt_array, 8, 0)
    # 保存可视化结果，输出图像将在 `workdirs/result.png` 路径下找到
    # show_result_pyplot(model,
    #                    img_path,
    #                    result,
    #                    show=False,
    #                    draw_gt=False,
    #                    out_file=f'out_dirs/{config_name}/{case_name}/{img_name}/pred.png',
    #                    metric='mDice:' + str(dice) + '%')
    # show_result_pyplot(model,
    #                    img_path,
    #                    result,
    #                    show=False,
    #                    draw_pred=False,
    #                    out_file=f'out_dirs/{config_name}/{case_name}/{img_name}/gt.png')
    return case_name, img_name, dice
dice_dict = {}
dice_list = []
progress_bar = ProgressBar(len(img_list))
for img_dir, ann_dir in zip(img_list, ann_list):
    case_name, img_name, dice = infer_on_image(img_dir, ann_dir)
    dice_dict.update(dict({img_name: dice}))
    dice_list.append(dice)
    progress_bar.update()
mdice = np.round(np.mean(dice_list), 2)
dice_dict.update({'mdice': mdice})
dump(dice_dict, f'out_dirs/{config_name}/infer_info.json')

# # 修改展示图像的时间，注意 0 是表示“无限”的特殊值
# vis_image = show_result_pyplot(model, img_path, result, wait_time=5)