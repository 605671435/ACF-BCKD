from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder, ModuleInputsRecorder
from mmseg.models.losses.kldiv_loss import KLDivLoss
from mmrazor.models.losses import KLDivergence
from razor.models.losses.kldiv_loss import KLDivergence3, CriterionKD

with read_base():
    from ..._base_.datasets.brats21 import *  # noqa
    from ..._base_.schedules.schedule_100e_sgd import *  # noqa
    from ..._base_.monai_runtime import *  # noqa
    # import teacher model and student model
    from ...unet.unetmod_base_d8_100e_sgd_brats21_96x96x96 import model as teacher_model  # noqa
    from ...unet.unetmod_tiny_d8_100e_sgd_brats21_96x96x96 import model as student_model  # noqa

teacher_ckpt = 'ckpts/unetmod_base_d8_100e_sgd_brats21_96x96x96/best_Dice_89-86_epoch_100.pth'  # noqa: E501
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
                # tau=4,
                loss_weight=10.0,
                # reduction='mean'
            )),
        student_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='segmentor')),
        teacher_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='segmentor')),
        loss_forward_mappings=dict(
            loss_kl=dict(
                preds_S=dict(from_student=True, recorder='logits'),
                preds_T=dict(from_student=False, recorder='logits')))))

find_unused_parameters = True

default_hooks.update(
    dict(
        logger=dict(interval=10, val_interval=50)))

val_cfg = dict(type=MonaiValLoop, print_log_per_case=False)
test_cfg = dict(type=MonaiTestLoop, print_log_per_case=False)

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='brats23', name='kd-unet-base-unet-tiny-50e'),
        )
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
