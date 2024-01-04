# last residual layer: segmentor.backbone.up_layers.2.0.conv2
from mmengine.config import read_base
from monai.losses import DiceLoss
from monai.networks.nets import SegResNet
from seg.models.segmentors.monai_model import MonaiSeg

with read_base():
    from .._base_.datasets.brats21 import *  # noqa
    from .._base_.schedules.schedule_50e_sgd import *  # noqa
    from .._base_.monai_runtime import *  # noqa

default_hooks.update(
    dict(
        logger=dict(interval=10, val_interval=50)))

val_cfg = dict(type=MonaiValLoop, print_log_per_case=False)
test_cfg = dict(type=MonaiTestLoop, print_log_per_case=False)

# model settings
model = dict(
    type=MonaiSeg,
    num_classes=3,
    roi_shapes=roi,
    backbone=dict(
        type=SegResNet,
        in_channels=4,
        init_filters=16,
        out_channels=3),
    loss_functions=dict(
        type=DiceLoss, to_onehot_y=False, sigmoid=True),
    infer_cfg=dict(
        inf_size=roi,
        sw_batch_size=4,    # number of sliding window batch size
        infer_overlap=0.5   # sliding window inference overlap
    ))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='brats23', name='segresnet-sgd-1000e'),
        define_metric_cfg=dict(Dice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
