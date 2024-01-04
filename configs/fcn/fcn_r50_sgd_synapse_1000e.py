from mmengine.config import read_base

with read_base():
    from .fcn_r18_sgd_synapse_1000e import *  # noqa

# model settings
model.update(
    dict(
        backbone=dict(depth=50),
        decoder=dict(
            in_channels=2048,
            channels=512)))
vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='fcn-r50-40k'),
        define_metric_cfg=dict(loss='min'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
