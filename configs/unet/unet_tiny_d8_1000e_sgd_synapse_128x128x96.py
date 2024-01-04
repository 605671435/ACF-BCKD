from mmengine.config import read_base

with read_base():
    from .unet_base_sgd_synapse import *   # noqa
    from .._base_.datasets.synapse_128x128x96 import *  # noqa

# model settings
model.update(
    dict(
        roi_shapes=roi,
        backbone=dict(
            type=UNet,
            channels=(32, 64, 128, 256),
            strides=(2, 2, 2),
            num_res_units=0),
        infer_cfg=dict(
            inf_size=roi,
            sw_batch_size=2)))

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
