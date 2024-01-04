from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder, ModuleInputsRecorder
from mmrazor.models.losses import FTLoss
from razor.models.architectures.connectors.factor_transfer_connectors import Translator, Paraphraser

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
            loss_ft=dict(
                type=FTLoss,
                loss_weight=1.0,
            )),
        student_recorders=dict(
            up_layer3=dict(type=ModuleInputsRecorder, source='segmentor.backbone.up_layer3')),
        teacher_recorders=dict(
            up_layer3=dict(type=ModuleInputsRecorder, source='segmentor.backbone.up_layer3')),
        connectors=dict(
            s_feat=dict(
                type=Translator, in_channel=64, out_channel=128),
            t_feat=dict(
                type=Paraphraser,
                phase='train',
                in_channel=128,
                out_channel=128)),
        loss_forward_mappings=dict(
            loss_ft=dict(
                s_feature=dict(from_student=True, recorder='up_layer3', connector='s_feat', data_idx=0),
                t_feature=dict(from_student=False, recorder='up_layer3', connector='t_feat', data_idx=0)))))

find_unused_parameters = True

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='ft-unet-base-unet-tiny-1000e'),
        )
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
