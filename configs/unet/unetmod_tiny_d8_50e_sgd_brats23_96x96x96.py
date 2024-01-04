from mmengine.config import read_base
from monai.losses import DiceCELoss
from seg.models.unet.monai_unet_mod import UNetMod
from seg.models.segmentors.monai_model import MonaiSeg

with read_base():
    from .._base_.datasets.brats23 import *  # noqa
    from .._base_.schedules.schedule_50e_sgd import *  # noqa
    from .._base_.monai_runtime import *  # noqa

# model settings
model = dict(
    type=MonaiSeg,
    num_classes=4,
    roi_shapes=roi,
    backbone=dict(
        type=UNetMod,
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=0),
    loss_functions=dict(
        type=DiceCELoss, to_onehot_y=True, softmax=True),
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
            project='brats23', name='unet-tiny-sgd-50e'),
        define_metric_cfg=dict(Dice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
