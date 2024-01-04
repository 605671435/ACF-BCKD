from mmengine.optim import CosineAnnealingLR
from torch.optim import AdamW
from mmengine.optim.scheduler import LinearLR
from mmengine.optim import AmpOptimWrapper
from mmengine.hooks import IterTimerHook, ParamSchedulerHook, DistSamplerSeedHook
from mmseg.engine.hooks import SegVisualizationHook
from seg.engine.hooks import MyCheckpointHook
from seg.engine.hooks.logger_hook import MyLoggerHook
from mmengine.runner.loops import EpochBasedTrainLoop
from seg.engine.runner.monai_loops import MonaiValLoop, MonaiTestLoop

train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=5000, val_interval=100)
val_cfg = dict(type=MonaiValLoop)
test_cfg = dict(type=MonaiTestLoop)
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=AdamW, lr=1e-4, weight_decay=1e-5))

param_scheduler = [
    # 在 [0, 100) 迭代时使用线性学习率
    dict(type=LinearLR,
         start_factor=1e-6,
         by_epoch=True,
         begin=0,
         end=50),
    # 在 [100, 900) 迭代时使用余弦学习率
    dict(type=CosineAnnealingLR,
         T_max=4950,
         by_epoch=True,
         begin=50,
         end=5000)]

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=MyLoggerHook, interval=10, val_interval=1),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=MyCheckpointHook,
                    by_epoch=True,
                    interval=100,
                    max_keep_ckpts=1,
                    save_best=['Dice'], rule='greater'),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook))