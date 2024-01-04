from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder, \
    ModuleInputsRecorder
from razor.models.losses.emkd_losses import PMD, IMD, RAD
# _stack_batch_gt
with read_base():
    from ..._base_.datasets.synapse import *  # noqa
    from ..._base_.schedules.schedule_1000e_sgd import *  # noqa
    from ..._base_.monai_runtime import *  # noqa
    # import teacher model and student model
    from ...unet.unet_base_sgd_synapse import model as teacher_model  # noqa
    from ...unet.unet_tiny_sgd_synapse import model as student_model  # noqa

teacher_ckpt = 'ckpts/unet50_sgd_synapse/best_Dice_83-29_epoch_780.pth'  # noqa: E501

model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=ConfigurableDistiller,
        student_recorders=dict(
            low_feat=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.model.0'),
            high_feat=dict(type=ModuleOutputsRecorder,
                           source='segmentor.backbone.model.1.submodule.1.submodule.0'),
            logits=dict(type=ModuleOutputsRecorder, source='segmentor'),
            gt=dict(type=ModuleInputsRecorder, source='loss_functions')),
        teacher_recorders=dict(
            low_feat=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.model.0'),
            high_feat=dict(type=ModuleOutputsRecorder,
                           source='segmentor.backbone.model.1.submodule.1.submodule.0'),
            logits=dict(type=ModuleOutputsRecorder, source='segmentor'),),
        distill_losses=dict(
            loss_rad_low=dict(type=RAD, num_classes=14, loss_weight=0.9),
            loss_rad_high=dict(type=RAD, num_classes=14, loss_weight=0.9),
        ),
        loss_forward_mappings=dict(
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
            project='synapse', name='emkd_unet_base_unet_tiny-1000e'),
        )
]
visualizer.update(
    dict(type=SegLocalVisualizer,
         vis_backends=vis_backends,
         name='visualizer'))
