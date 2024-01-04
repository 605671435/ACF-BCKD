from mmengine.config import read_base
from monai.losses import DiceCELoss
from seg.models.backbones import ResNet
from seg.models.decode_heads import FCNHead
from seg.models.segmentors.monai_model import MonaiSeg

with read_base():
    from .._base_.datasets.synapse import *  # noqa
    from .._base_.schedules.schedule_1000e_sgd import *  # noqa
    from .._base_.monai_runtime import *  # noqa

# model settings
model = dict(
    type=MonaiSeg,
    num_classes=14,
    roi_shapes=roi,
    backbone=dict(
        type=ResNet,
        depth=18,
        conv_cfg=dict(type='Conv3d'),
        norm_cfg=dict(type='IN3d', requires_grad=True),
        in_channels=1),
    decoder=dict(
        type=FCNHead,
        in_channels=512,
        channels=128,
        conv_cfg=dict(type='Conv3d'),
        norm_cfg=dict(type='IN3d', requires_grad=True),
        num_classes=14),
    loss_functions=dict(
        type=DiceCELoss, to_onehot_y=True, softmax=True),
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
            project='synapse', name='swin-unetr-40k'),
        define_metric_cfg=dict(loss='min'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
