from mmengine.config import read_base
from seg.models.unet.monai_unet import UNetKD
with read_base():
    from .unet_base_sgd_synapse import *   # noqa

# model settings
model.update(
    dict(
        backbone=dict(
            type=UNetKD,
            channels=(64, 128, 256, 512),
            strides=(2, 2, 2),
            num_res_units=0)))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='unet-tiny-1000e'),
        define_metric_cfg=dict(loss='min'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
