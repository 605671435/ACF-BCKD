from mmengine.config import read_base
from monai.networks.nets import AttentionUnet

with read_base():
    from ..unet.unet_base_sgd_synapse import *  # noqa

# model settings
model['model_cfg'] = dict(
    type=AttentionUnet,
    spatial_dims=3,
    in_channels=1,
    out_channels=14,
    channels=(2, 4, 8),
    strides=(2, 2))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='attn-unet50-40k'),
        define_metric_cfg=dict(loss='min'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
