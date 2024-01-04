# last residual layer: segmentor.backbone.up_layers.2.0.conv2
from mmengine.config import read_base
from monai.losses import DiceLoss
from monai.networks.nets import SegResNetVAE
from seg.models.segmentors.monai_model import MonaiSeg

with read_base():
    from .._base_.datasets.brats21 import *  # noqa
    from .._base_.schedules.schedule_50e_sgd import *  # noqa
    from .._base_.monai_runtime import *  # noqa

# model settings
model = dict(
    type=MonaiSeg,
    num_classes=14,
    roi_shapes=roi,
    vae_loss=True,
    vae_loss_weight=0.1,
    backbone=dict(
        type=SegResNetVAE,
        input_image_size=[96] * 3,
        in_channels=4,
        out_channels=3),
    loss_functions=dict(
        type=DiceLoss, to_onehot_y=False, sigmoid=True),
    infer_cfg=dict(
        inf_size=roi,
        sw_batch_size=4,    # number of sliding window batch size
        infer_overlap=0.5   # sliding window inference overlap
    ))

default_hooks.update(
    dict(
        logger=dict(interval=5, val_interval=50)))

val_cfg = dict(type=MonaiValLoop, print_log_per_case=False)
test_cfg = dict(type=MonaiTestLoop, print_log_per_case=False)

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='lcov-sgd-1000e'),
        define_metric_cfg=dict(loss='min'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
