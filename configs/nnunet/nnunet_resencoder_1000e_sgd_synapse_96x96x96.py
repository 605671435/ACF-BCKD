from mmengine.config import read_base
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from seg.models.segmentors.monai_model import MonaiSeg
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from torch.nn import Conv3d, InstanceNorm3d, LeakyReLU
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
        type=ResidualEncoderUNet,
        input_channels=1,
        n_stages=5,
        features_per_stage=(32, 64, 128, 256, 512),
        conv_op=Conv3d,
        kernel_sizes=3,
        strides=(1, 2, 2, 2, 2),
        n_blocks_per_stage=(2, 2, 2, 2, 2),
        n_conv_per_stage_decoder=(2, 2, 2, 2),
        conv_bias=True,
        num_classes=14,
        norm_op=InstanceNorm3d,
        norm_op_kwargs={},
        dropout_op=None,
        nonlin=LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=False),
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
            project='synapse', name='nnunet-seg-1000e'),
        define_metric_cfg=dict(loss='min'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
