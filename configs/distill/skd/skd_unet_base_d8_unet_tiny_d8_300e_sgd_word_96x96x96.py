from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.architectures.connectors import ConvModuleConnector
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder, ModuleInputsRecorder
from razor.models.losses.skd import CriterionPairWiseforWholeFeatAfterPool, CriterionPixelWise

with read_base():
    from ..._base_.datasets.word import *  # noqa
    from ..._base_.schedules.schedule_300e_sgd import *  # noqa
    from ..._base_.monai_runtime import *  # noqa
    # import teacher model and student model
    from ...unet.unetmod_base_d8_300e_sgd_word_96x96x96 import model as teacher_model  # noqa
    from ...unet.unetmod_tiny_d8_300e_sgd_word_96x96x96 import model as student_model  # noqa

teacher_ckpt = 'ckpts/unetmod_base_d8_300e_sgd_word_96x96x96/best_Dice_84-12_epoch_300.pth'  # noqa: E501
model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=ConfigurableDistiller,
        distill_losses=dict(
            loss_pi=dict(
                type=CriterionPixelWise,
                loss_weight=10.0),
            loss_pa=dict(
                type=CriterionPairWiseforWholeFeatAfterPool,
                loss_weight=1.0),
        ),
        student_recorders=dict(
            feat=dict(type=ModuleInputsRecorder, source='segmentor.backbone.up_layer3'),
            logits=dict(type=ModuleOutputsRecorder, source='segmentor'),
        ),
        teacher_recorders=dict(
            feat=dict(type=ModuleInputsRecorder, source='segmentor.backbone.up_layer3'),
            logits=dict(type=ModuleOutputsRecorder, source='segmentor'),
        ),
        loss_forward_mappings=dict(
            loss_pi=dict(
                preds_S=dict(from_student=True, recorder='logits'),
                preds_T=dict(from_student=False, recorder='logits')),
            loss_pa=dict(
                feat_S=dict(from_student=True, recorder='feat', data_idx=0),
                feat_T=dict(from_student=False, recorder='feat', data_idx=0),
            ),
        )))

find_unused_parameters = True

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='word', name='skd-unet-base-unet-tiny-200e'),
        )
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
