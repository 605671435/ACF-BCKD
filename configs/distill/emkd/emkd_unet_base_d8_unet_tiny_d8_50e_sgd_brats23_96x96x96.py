from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder, \
    ModuleInputsRecorder
from razor.models.losses.emkd_losses import IMD, RAD_BraTS23
from razor.models.losses.kldiv_loss import CriterionKD
# _stack_batch_gt
with read_base():
    from ..._base_.datasets.brats21 import *  # noqa
    from ..._base_.schedules.schedule_50e_sgd import *  # noqa
    from ..._base_.monai_runtime import *  # noqa
    # import teacher model and student model
    from ...unet.unetmod_base_d8_100e_sgd_brats21_96x96x96 import model as teacher_model  # noqa
    from ...unet.unetmod_tiny_d8_50e_sgd_brats21_96x96x96 import model as student_model  # noqa

default_hooks.update(
    dict(
        logger=dict(interval=10, val_interval=50)))

val_cfg = dict(type=MonaiValLoop, print_log_per_case=False)
test_cfg = dict(type=MonaiTestLoop, print_log_per_case=False)

teacher_ckpt = 'ckpts/unetmod_base_d8_100e_sgd_brats21_96x96x96/best_Dice_89-86_epoch_100.pth'  # noqa: E501

model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=ConfigurableDistiller,
        student_recorders=dict(
            low_feat=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.down_layer1'),
            high_feat=dict(type=ModuleOutputsRecorder,
                           source='segmentor.backbone.down_layer3'),
            logits=dict(type=ModuleOutputsRecorder, source='segmentor'),
            gt=dict(type=ModuleInputsRecorder, source='loss_functions')),
        teacher_recorders=dict(
            low_feat=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.down_layer1'),
            high_feat=dict(type=ModuleOutputsRecorder,
                           source='segmentor.backbone.down_layer3'),
            logits=dict(type=ModuleOutputsRecorder, source='segmentor'),),
        distill_losses=dict(
            loss_kl=dict(type=CriterionKD, sigmoid=True, loss_weight=0.1),
            loss_imd_low=dict(type=IMD, loss_weight=0.9),
            loss_imd_high=dict(type=IMD, loss_weight=0.9),
            loss_rad_low=dict(type=RAD_BraTS23, num_classes=3, loss_weight=0.9),
            loss_rad_high=dict(type=RAD_BraTS23, num_classes=3, loss_weight=0.9),
        ),
        loss_forward_mappings=dict(
            loss_kl=dict(
                preds_S=dict(from_student=True, recorder='logits'),
                preds_T=dict(from_student=False, recorder='logits')),
            loss_imd_low=dict(
                s_feature=dict(from_student=True, recorder='low_feat'),
                t_feature=dict(from_student=False, recorder='low_feat')),
            loss_imd_high=dict(
                s_feature=dict(from_student=True, recorder='high_feat'),
                t_feature=dict(from_student=False, recorder='high_feat')),
            loss_rad_low=dict(
                logits_S=dict(from_student=True, recorder='low_feat'),
                logits_T=dict(from_student=False, recorder='low_feat'),
                ground_truth=dict(from_student=True, recorder='gt', data_idx=1)),
            loss_rad_high=dict(
                logits_S=dict(from_student=True, recorder='high_feat'),
                logits_T=dict(from_student=False, recorder='high_feat'),
                ground_truth=dict(from_student=True, recorder='gt', data_idx=1))
            )))

find_unused_parameters = True

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='emkd_unet_base_unet_small-1000e'),
        )
]
visualizer.update(
    dict(type=SegLocalVisualizer,
         vis_backends=vis_backends,
         name='visualizer'))
