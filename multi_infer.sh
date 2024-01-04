#!/usr/bin/env bash

CONFIGS=(
'configs/dsa/fcn_r50_ex_a_40k_synapse.py'
'configs/dsa/fcn_r50_ex_b_40k_synapse.py'
'configs/gcnet/fcn_r50_gcb_40k_synapse.py'
'configs/hamnet/fcn_r50_hamnet_40k_synapse.py'
'configs/psa/fcn_r50_psa_40k_synapse.py'
'configs/se/fcn_r50_se_40k_synapse.py'
'configs/dsa/unet_r50_ex_b_40k_synapse.py'
'configs/ecanet/fcn_r50_ecanet_40k_synapse.py'
'configs/cbam/fcn_r50_cbam_40k_synapse.py'
'configs/fcn/fcn_r50_d8_40k_synapse.py'
'configs/ccnet/fcn_r50_ccnet_40k_synapse.py'
'configs/fcanet/fcn_fcanet50_40k_synapse.py'
)
WORK_DIRS=(
'work_dirs_026/fcn_r50_ex_a_40k_synapse/3-run_20230904_003511'
'work_dirs_026/fcn_r50_ex_b_40k_synapse/3-run_20230904_051602'
'work_dirs_026/fcn_r50_gcb_40k_synapse/3-run_20230904_004735'
'work_dirs_026/fcn_r50_hamnet_40k_synapse/3-run_20230908_002940'
'work_dirs_026/fcn_r50_psa_40k_synapse/3-run_20230913_184046'
'work_dirs_026/fcn_r50_se_40k_synapse/3-run_20230913_162148'
'work_dirs_026/unet_r50_ex_b_40k_synapse/3-run_20230905_042505'
'work_dirs_026/fcn_r50_ecanet_40k_synapse/3-run_20230904_004626'
'work_dirs_026/fcn_r50_cbam_40k_synapse/3-run_20230912_181311'
'work_dirs_026/fcn_r50_d8_40k_synapse/3-run_20230904_105704'
'work_dirs_026/fcn_r50_ccnet_40k_synapse/3-run_20230913_223519'
'work_dirs_026/fcn_fcanet50_40k_synapse/3-run_20230904_004703'
)
#IMGS=(
#'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0001/case0001_slice094.jpg'
#'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0002/case0002_slice087.jpg'
#'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0004/case0004_slice106.jpg'
#'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0022/case0022_slice063.jpg'
#'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0036/case0036_slice130.jpg'
#)
IMGS=(
'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0001/case0001_slice094.jpg'
'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0002/case0002_slice087.jpg'
'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0004/case0004_slice106.jpg'
'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0022/case0022_slice063.jpg'
'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0036/case0036_slice130.jpg'
'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0001/case0001_slice112.jpg'
'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0002/case0002_slice110.jpg'
'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0022/case0022_slice066.jpg'
'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0032/case0032_slice108.jpg'
'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0036/case0036_slice138.jpg'
)
# 使用 for in 循环遍历数组元素
for IMG in "${IMGS[@]}"
do
#  PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#    python tools/visualizations/vis_gt.py $IMG
  i=0
  for CONFIG in "${CONFIGS[@]}"
  do
    echo $CONFIG
    echo ${WORK_DIRS[$i]}
    echo $IMG
    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    python tools/inference.py \
      $CONFIG \
      ${WORK_DIRS[$i]} \
      $IMG
    i=$i+1
  done
done
