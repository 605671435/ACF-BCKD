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
    from ..._base_.datasets.synapse import *  # noqa
    from ..._base_.schedules.schedule_1000e_sgd import *  # noqa
    from ..._base_.monai_runtime import *  # noqa
    # import teacher model and student model
    from ...unet.unetmod_base_d8_1000e_sgd_synapse_96x96x96 import model as teacher_model  # noqa
    from ...mobile_unet.mobileunet_1000e_sgd_synapse_96x96x96 import model as student_model  # noqa

teacher_ckpt = 'ckpts/unetmod_base_d8_1000e_sgd_synapse_96x96x96/best_Dice_81-69_epoch_800.pth'  # noqa: E501
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
                loss_weight=1.0,
            ),
            loss_hd=dict(
                type=LogHausdorffDTLoss,
                loss_weight=0.,
            ),
            loss_at6=dict(
                type=EXKDV2_Loss,
                spatial_dims=3,
                alpha=1.0,
                beta=1.0,
                student_channels=16,
                teacher_channels=64,
                student_shape=96,
                teacher_shape=48,
                loss_weight=1.0),
        ),
        student_recorders=dict(
            up_conv2=dict(
                type=ModuleOutputsRecorder,
                source='segmentor.backbone.DConv4x4'),
            logits=dict(type=ModuleOutputsRecorder, source='segmentor'),
            gt_labels=dict(type=ModuleInputsRecorder, source='loss_functions')
        ),
        teacher_recorders=dict(
            up2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layer2.1'),
            up_ru2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layer2.1.conv'),
            logits=dict(type=ModuleOutputsRecorder, source='segmentor')
        ),
        loss_forward_mappings=dict(
            loss_at6=dict(
                s_feature=dict(from_student=True, recorder='up_conv2'),
                t_feature=dict(from_student=False, recorder='up2'),
                t_residual=dict(from_student=False, recorder='up_ru2')),
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
         # eta_min=0.5, gamma=0.0005
         )
)

find_unused_parameters = True

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='kd-unet-base-unet-tiny-1000e'),
        )
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
