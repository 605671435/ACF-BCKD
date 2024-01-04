from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.architectures.connectors import TorchFunctionalConnector
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder, ModuleInputsRecorder
from razor.models.losses.kldiv_loss import CriterionKD
from razor.models.losses.ifvd_loss import CriterionIFV

with read_base():
    from ..._base_.datasets.synapse import *  # noqa
    from ..._base_.schedules.schedule_1000e_sgd import *  # noqa
    from ..._base_.monai_runtime import *  # noqa
    # import teacher model and student model
    from ...swin_unetr.swinunetr_base_5000e_synapse import model as teacher_model  # noqa
    from ...unet.unetmod_tiny_d8_1000e_sgd_synapse_96x96x96 import model as student_model  # noqa

teacher_ckpt = 'ckpts/swin_unetr.base_5000ep_f48_lr2e-4_pretrained_mmengine.pth'  # noqa: E501
model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=ConfigurableDistiller,
        distill_losses=dict(
            loss_kl=dict(
                type=CriterionKD,
                loss_weight=10.0),
            loss_ifv=dict(
                type=CriterionIFV,
                classes=14,
                loss_weight=200.0),
        ),
        student_recorders=dict(
            up_layer2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layer2'),
            logits=dict(type=ModuleOutputsRecorder, source='segmentor'),
            gt_labels=dict(type=ModuleInputsRecorder, source='loss_functions')
        ),
        teacher_recorders=dict(
            up_layer2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.decoder1'),
            logits=dict(type=ModuleOutputsRecorder, source='segmentor'),
        ),
        connectors=dict(
            up_layer2=dict(
                type=TorchFunctionalConnector,
                function_name='interpolate',
                func_args=dict(size=48, mode='nearest'))
        ),
        loss_forward_mappings=dict(
            loss_ifv=dict(
                feat_S=dict(from_student=True, recorder='up_layer2'),
                feat_T=dict(from_student=False, recorder='up_layer2', connector='up_layer2'),
                target=dict(from_student=True, recorder='gt_labels', data_idx=1)),
            loss_kl=dict(
                preds_S=dict(from_student=True, recorder='logits'),
                preds_T=dict(from_student=False, recorder='logits')),
        )))

find_unused_parameters = True

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='ifvd-swinunetr-base-unet-tiny-1000e'),
        )
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
