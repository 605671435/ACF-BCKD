from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder, ModuleInputsRecorder
from mmrazor.models.losses import ATLoss

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
            loss_at6=dict(
                type=ATLoss,
                loss_weight=25000.0),
        ),
        student_recorders=dict(
            up_layer2=dict(
                type=ModuleOutputsRecorder,
                source='segmentor.backbone.up_layer2'),),
        teacher_recorders=dict(
            up_layer2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layer2'),),
        loss_forward_mappings=dict(
            loss_at6=dict(
                s_feature=dict(from_student=True, recorder='up_layer2'),
                t_feature=dict(from_student=False, recorder='up_layer2')),
        )))

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
