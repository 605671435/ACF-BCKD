from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.architectures.connectors import ConvModuleConnector
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder, ModuleInputsRecorder
from razor.models.losses.cirkd_loss import CriterionKD, StudentSegContrast, CriterionMiniBatchCrossImagePair


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
            loss_minibatch=dict(
                type=CriterionMiniBatchCrossImagePair,
                temperature=1,
                pooling=True,
                loss_weight=1.0),
            loss_memory=dict(
                type=StudentSegContrast,
                num_classes=14,
                s_channels=64,
                t_channels=128,
                region_contrast_size=96,
                pixel_contrast_size=96,
                loss_weight=1.0),
            loss_kl=dict(
                type=CriterionKD,
                loss_weight=1.0),
        ),
        student_recorders=dict(
            feat=dict(type=ModuleInputsRecorder, source='segmentor.backbone.up_layer3'),
            logits=dict(type=ModuleOutputsRecorder, source='segmentor'),
            gt_labels=dict(type=ModuleInputsRecorder, source='loss_functions')
        ),
        teacher_recorders=dict(
            feat=dict(type=ModuleInputsRecorder, source='segmentor.backbone.up_layer3'),
            logits=dict(type=ModuleOutputsRecorder, source='segmentor'),
        ),
        loss_forward_mappings=dict(
            loss_minibatch=dict(
                feat_S=dict(from_student=True, recorder='feat', data_idx=0),
                feat_T=dict(from_student=False, recorder='feat', data_idx=0)),
            loss_memory=dict(
                s_feats=dict(from_student=True, recorder='feat', data_idx=0),
                t_feats=dict(from_student=False, recorder='feat', data_idx=0),
                predict=dict(from_student=True, recorder='logits'),
                labels=dict(from_student=True, recorder='gt_labels', data_idx=1)),
            loss_kl=dict(
                pred=dict(from_student=True, recorder='logits'),
                soft=dict(from_student=False, recorder='logits')),
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
