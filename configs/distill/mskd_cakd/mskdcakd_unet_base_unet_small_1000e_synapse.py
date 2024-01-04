from mmengine.config import read_base

with read_base():
    from .mskdcakd_unet_base_unet_tiny_1000e_synapse import *   # noqa
    # import teacher model and student model
    from ...unet.unet_base_sgd_synapse import model as teacher_model  # noqa
    from ...unet.unet_small_sgd_synapse import model as student_model  # noqa

teacher_ckpt = 'ckpts/unet50_sgd_synapse/best_Dice_83-29_epoch_780.pth'  # noqa: E501

model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=ConfigurableDistiller,
        student_recorders=dict(
            features=dict(type=ModuleInputsRecorder,
                          source='segmentor.backbone.model.2.0'),
            outputs=dict(type=ModuleOutputsRecorder,
                         source='segmentor')),
        teacher_recorders=dict(
            features=dict(type=ModuleInputsRecorder,
                          source='segmentor.backbone.model.2.0'),
            outputs=dict(type=ModuleOutputsRecorder,
                         source='segmentor')),
        distill_losses=dict(
            loss_mskdcakd=dict(type=MSKDCAKDLoss, loss_weight=0.25),
        ),
        loss_forward_mappings=dict(
            loss_mskdcakd=dict(
                student_outputs=dict(from_student=True, recorder='outputs'),
                teacher_outputs=dict(from_student=False, recorder='outputs'),
                student_features=dict(from_student=True, recorder='features', data_idx=0),
                teacher_features=dict(from_student=False, recorder='features', data_idx=0),
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
