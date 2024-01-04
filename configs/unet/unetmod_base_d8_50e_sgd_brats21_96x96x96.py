from mmengine.config import read_base
with read_base():
    from .unetmod_tiny_d8_50e_sgd_brats21_96x96x96 import *   # noqa

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
            project='brats21', name='unet-base-50e'),
        define_metric_cfg=dict(Dice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
