from mmengine.config import read_base
from seg.models.unet.monai_unet_mod import UNetMod
with read_base():
    from ..unet.unetmod_tiny_d8_1000e_sgd_synapse_96x96x96 import *   # noqa
    from .._base_.datasets.synapse import *  # noqa

# model settings
model.update(
    dict(
        backbone=dict(attn_module=True)))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='unet-small-1000e'),
        define_metric_cfg=dict(loss='min'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
