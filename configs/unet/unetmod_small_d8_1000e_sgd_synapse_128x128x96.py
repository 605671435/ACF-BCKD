from mmengine.config import read_base
from seg.models.unet.monai_unet_mod import UNetMod
with read_base():
    from .unet_base_sgd_synapse import *   # noqa
    from .._base_.datasets.synapse_128x128x96 import *  # noqa

# model settings
model.update(
    dict(
        roi_shapes=roi,
        backbone=dict(
            type=UNetMod,
            channels=(32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=1),
        infer_cfg=dict(
            inf_size=roi)))

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
