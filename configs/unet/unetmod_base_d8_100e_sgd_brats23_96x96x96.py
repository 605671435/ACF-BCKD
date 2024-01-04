from mmengine.config import read_base
with read_base():
    from .unetmod_tiny_d8_50e_sgd_brats23_96x96x96 import *   # noqa
    from .._base_.schedules.schedule_100e_sgd import * # noqa

default_hooks.update(
    dict(
        logger=dict(interval=50, val_interval=50)))

val_cfg = dict(type=MonaiValLoop, print_log_per_case=False)
test_cfg = dict(type=MonaiTestLoop, print_log_per_case=False)

# model settings
model.update(
    dict(
        backbone=dict(
            channels=(64, 128, 256, 512),
            strides=(2, 2, 2),
            num_res_units=2)))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='brats23', name='unet-base-100e'),
        define_metric_cfg=dict(Dice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
