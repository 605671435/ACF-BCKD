from mmengine.config import read_base

with read_base():
    from .unet_base_sgd_synapse import *   # noqa
    from .._base_.schedules.schedule_500e_sgd import *  # noqa

# model settings
model.update(
    dict(
        backbone=dict(
            type=UNet,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=0)))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='unet-tiny-d16-sgd-500e'),
        define_metric_cfg=dict(Dice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
