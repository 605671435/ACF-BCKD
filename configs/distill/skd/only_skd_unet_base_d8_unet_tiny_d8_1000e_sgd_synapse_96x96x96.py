from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.architectures.connectors import ConvModuleConnector
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder, ModuleInputsRecorder
from razor.models.losses.skd import CriterionPairWiseforWholeFeatAfterPool

with read_base():
    from ..._base_.datasets.synapse import *  # noqa
    from ..._base_.schedules.schedule_1000e_sgd import *  # noqa
    from ..._base_.monai_runtime import *  # noqa
    # import teacher model and student model
    from ...unet.unetmod_base_d8_1000e_sgd_synapse_96x96x96 import model as teacher_model  # noqa
    from ...unet.unetmod_tiny_d8_1000e_sgd_synapse_96x96x96 import model as student_model  # noqa

teacher_ckpt = 'ckpts/unetmod_base_d8_1000e_sgd_synapse_96x96x96/best_Dice_81-69_epoch_800.pth'  # noqa: E501
model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=ConfigurableDistiller,
        distill_losses=dict(
            loss_skd=dict(
                type=CriterionPairWiseforWholeFeatAfterPool,
                loss_weight=1.0),
        ),
        student_recorders=dict(
            bottom=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.bottom_layer'),
        ),
        teacher_recorders=dict(
            bottom=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.bottom_layer'),
        ),
        connectors=dict(
            loss_s1_sfeat=dict(type=ConvModuleConnector,
                               conv_cfg=dict(type='Conv3d'),
                               in_channel=256,
                               out_channel=512),
        ),
        loss_forward_mappings=dict(
            loss_skd=dict(
                feat_S=dict(from_student=True, recorder='bottom',
                            connector='loss_s1_sfeat'
                            ),
                feat_T=dict(from_student=False, recorder='bottom')),
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
