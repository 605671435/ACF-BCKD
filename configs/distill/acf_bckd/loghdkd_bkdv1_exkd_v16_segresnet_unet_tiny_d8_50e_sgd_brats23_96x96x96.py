# only at6
# residual to conv
from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder, ModuleInputsRecorder
from razor.models.losses.exkd_loss import EXKDV2_Loss
from razor.models.losses.boundary_loss import BoundaryKDV1
from razor.models.losses.hd_loss import LogHausdorffDTLoss
from seg.engine.hooks.schedule_hook import DistillLossWeightScheduleHook

with read_base():
    from ..._base_.datasets.brats21 import *  # noqa
    from ..._base_.schedules.schedule_50e_sgd import *  # noqa
    from ..._base_.monai_runtime import *  # noqa
    # import teacher model and student model
    from ...segresnet.segresnet_100e_sgd_brats23_96x96x96 import model as teacher_model  # noqa
    from ...unet.unetmod_tiny_d8_50e_sgd_brats21_96x96x96 import model as student_model  # noqa

default_hooks.update(
    dict(
        logger=dict(interval=10, val_interval=50)))

val_cfg = dict(type=MonaiValLoop, print_log_per_case=False)
test_cfg = dict(type=MonaiTestLoop, print_log_per_case=False)

teacher_ckpt = 'ckpts/segresnet_100e_sgd_brats23_96x96x96/best_Dice_90-08_epoch_100.pth'  # noqa: E501
model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=ConfigurableDistiller,
        distill_losses=dict(
            loss_boundary=dict(
                type=BoundaryKDV1,
                num_classes=3,
                include_background=True,
                one_hot_target=False,
                loss_weight=1.0,
            ),
            loss_hd=dict(
                type=LogHausdorffDTLoss,
                include_background=True,
                to_onehot_y=False,
                sigmoid=True,
                softmax=False,
                loss_weight=0.,
            ),
            loss_at6=dict(
                type=EXKDV2_Loss,
                spatial_dims=3,
                alpha=1.0,
                beta=1.0,
                student_channels=32,
                teacher_channels=16,
                student_shape=48,
                teacher_shape=96,
                loss_weight=1.0),
        ),
        student_recorders=dict(
            output=dict(
                type=ModuleOutputsRecorder,
                source='segmentor.backbone.up_layer2.conv'),
            logits=dict(type=ModuleOutputsRecorder, source='segmentor'),
            gt_labels=dict(type=ModuleInputsRecorder, source='loss_functions')
        ),
        teacher_recorders=dict(
            output=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layers.2.0'),
            ru=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layers.2.0.conv2'),
            logits=dict(type=ModuleOutputsRecorder, source='segmentor')
        ),
        loss_forward_mappings=dict(
            loss_at6=dict(
                s_feature=dict(from_student=True, recorder='output'),
                t_feature=dict(from_student=False, recorder='output'),
                t_residual=dict(from_student=False, recorder='ru')),
            loss_boundary=dict(
                preds_S=dict(from_student=True, recorder='logits'),
                preds_T=dict(from_student=False, recorder='logits'),
                gt_labels=dict(
                    recorder='gt_labels', from_student=True, data_idx=1),
            ),
            loss_hd=dict(
                preds_S=dict(from_student=True, recorder='logits'),
                preds_T=dict(from_student=False, recorder='logits'),
                target=dict(
                    recorder='gt_labels', from_student=True, data_idx=1),
            ))
        ))

custom_hooks.append(
    dict(type=DistillLossWeightScheduleHook,
         eta_min=0.5,
         gamma=0.01
         )
)


find_unused_parameters = True

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='exkd-swinunetr-base-unet-tiny-1000e'),
        )
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
