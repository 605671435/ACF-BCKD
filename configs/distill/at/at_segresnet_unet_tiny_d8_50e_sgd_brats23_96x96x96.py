from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder, ModuleInputsRecorder
from razor.models.losses.at_loss import ATLoss

with read_base():
    from ..._base_.datasets.brats21 import *  # noqa
    from ..._base_.schedules.schedule_50e_sgd import *  # noqa
    from ..._base_.monai_runtime import *  # noqa
    # import teacher model and student model
    from ...segresnet.segresnet_100e_sgd_brats23_96x96x96 import model as teacher_model  # noqa
    from ...unet.unetmod_tiny_d8_50e_sgd_brats21_96x96x96 import model as student_model  # noqa

default_hooks.update(
    dict(
        logger=dict(interval=10, val_interval=50)))

val_cfg = dict(type=MonaiValLoop, print_log_per_case=False)
test_cfg = dict(type=MonaiTestLoop, print_log_per_case=False)

teacher_ckpt = 'ckpts/segresnet_100e_sgd_brats23_96x96x96/best_Dice_90-08_epoch_100.pth'  # noqa: E501
model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=ConfigurableDistiller,
        distill_losses=dict(
            loss_at1=dict(
                type=ATLoss,
                loss_weight=200.0),
            loss_at2=dict(
                type=ATLoss,
                loss_weight=200.0),
            loss_at3=dict(
                type=ATLoss,
                loss_weight=200.0),
            loss_at4=dict(
                type=ATLoss,
                loss_weight=200.0),
            loss_at5=dict(
                type=ATLoss,
                loss_weight=200.0),
            loss_at6=dict(
                type=ATLoss,
                loss_weight=200.0),
            loss_at7=dict(
                type=ATLoss,
                loss_weight=200.0),
        ),
        student_recorders=dict(
            down_layer1=dict(
                type=ModuleOutputsRecorder,
                source='segmentor.backbone.down_layer1'),
            down_layer2=dict(
                type=ModuleOutputsRecorder,
                source='segmentor.backbone.down_layer2'),
            down_layer3=dict(
                type=ModuleOutputsRecorder,
                source='segmentor.backbone.down_layer3'),
            bottom_layer=dict(
                type=ModuleOutputsRecorder,
                source='segmentor.backbone.bottom_layer'),
            up_layer1=dict(
                type=ModuleOutputsRecorder,
                source='segmentor.backbone.up_layer1'),
            up_layer2=dict(
                type=ModuleOutputsRecorder,
                source='segmentor.backbone.up_layer2'),
            up_layer3=dict(
                type=ModuleOutputsRecorder,
                source='segmentor.backbone.up_layer3')),
        teacher_recorders=dict(
            down_layer1=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.down_layers.0'),
            down_layer2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.down_layers.1'),
            down_layer3=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.down_layers.2'),
            bottom_layer=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.down_layers.3'),
            up_layer1=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layers.0'),
            up_layer2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layers.1'),
            up_layer3=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layers.2')),
        loss_forward_mappings=dict(
            loss_at1=dict(
                s_feature=dict(from_student=True, recorder='down_layer1'),
                t_feature=dict(from_student=False, recorder='down_layer1')),
            loss_at2=dict(
                s_feature=dict(from_student=True, recorder='down_layer2'),
                t_feature=dict(from_student=False, recorder='down_layer2')),
            loss_at3=dict(
                s_feature=dict(from_student=True, recorder='down_layer3'),
                t_feature=dict(from_student=False, recorder='down_layer3')),
            loss_at4=dict(
                s_feature=dict(from_student=True, recorder='bottom_layer'),
                t_feature=dict(from_student=False, recorder='bottom_layer')),
            loss_at5=dict(
                s_feature=dict(from_student=True, recorder='up_layer1'),
                t_feature=dict(from_student=False, recorder='up_layer1')),
            loss_at6=dict(
                s_feature=dict(from_student=True, recorder='up_layer2'),
                t_feature=dict(from_student=False, recorder='up_layer2')),
            loss_at7=dict(
                s_feature=dict(from_student=True, recorder='up_layer3'),
                t_feature=dict(from_student=False, recorder='up_layer3')),
        )))

find_unused_parameters = True

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='brats23', name='at-unet-base-unet-tiny-50e'),
        )
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
