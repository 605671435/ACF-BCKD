from mmengine.config import read_base
from monai.losses import DiceCELoss
from monai.networks.nets import SwinUNETR
from seg.models.segmentors.monai_model import MonaiModel

with read_base():
    from .._base_.datasets.synapse import *  # noqa
    from .._base_.schedules.schedule_5000e_adamw import *  # noqa
    from .._base_.monai_runtime import *  # noqa

roi = [64, 64, 64]

dataloader_cfg.update(
    dict(
        workers=1,
        roi_x=roi[0],
        roi_y=roi[1],
        roi_z=roi[2]))
# model settings
model = dict(
    type=MonaiModel,
    num_classes=14,
    model_cfg=dict(
        type=SwinUNETR,
        img_size=roi,
        feature_size=12,
        in_channels=1,
        out_channels=14,
        spatial_dims=3),
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
