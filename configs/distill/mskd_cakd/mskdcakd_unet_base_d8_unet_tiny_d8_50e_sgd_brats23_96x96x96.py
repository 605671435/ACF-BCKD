from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder, \
    ModuleInputsRecorder
from razor.models.losses.mskd_cakd import MSKDCAKDLoss

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
            features=dict(type=ModuleOutputsRecorder,
                          source='segmentor.backbone.up_layer2'),
            outputs=dict(type=ModuleOutputsRecorder,
                         source='segmentor.backbone.up_layer3')),
        teacher_recorders=dict(
            features=dict(type=ModuleOutputsRecorder,
                          source='segmentor.backbone.up_layer2'),
            outputs=dict(type=ModuleOutputsRecorder,
                         source='segmentor.backbone.up_layer3')),
        distill_losses=dict(
            loss_mskdcakd=dict(type=MSKDCAKDLoss, sigmoid=True, loss_weight=0.25),
        ),
        loss_forward_mappings=dict(
            loss_mskdcakd=dict(
                student_outputs=dict(from_student=True, recorder='outputs'),
                teacher_outputs=dict(from_student=False, recorder='outputs'),
                student_features=dict(from_student=True, recorder='features'),
                teacher_features=dict(from_student=False, recorder='features'),
            ))))

find_unused_parameters = True

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='mskdcakd_unet_base_unet_small-1000e'),
        )
]
visualizer.update(
    dict(type=SegLocalVisualizer,
         vis_backends=vis_backends,
         name='visualizer'))
