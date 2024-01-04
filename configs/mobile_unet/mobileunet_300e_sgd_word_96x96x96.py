from mmengine.config import read_base
from monai.losses import DiceCELoss
from seg.models.nets.mobile_unet import MobileUNet
from seg.models.segmentors.monai_model import MonaiSeg

with read_base():
    from .._base_.datasets.word import *  # noqa
    from .._base_.schedules.schedule_300e_sgd import *  # noqa
    from .._base_.monai_runtime import *  # noqa

# model settings
model = dict(
    type=MonaiSeg,
    num_classes=17,
    roi_shapes=roi,
    backbone=dict(
        type=MobileUNet,
        in_channels=1,
        num_classes=17),
    loss_functions=dict(
        type=DiceCELoss, to_onehot_y=True, softmax=True),
    infer_cfg=dict(
        inf_size=roi,
        sw_batch_size=1,    # number of sliding window batch size
        infer_overlap=0.5   # sliding window inference overlap
    ))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='word', name='mobileunet-sgd-300e'),
        define_metric_cfg=dict(Dice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
