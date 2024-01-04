from mmengine.config import read_base

with read_base():
    from .cwd_swinunetr_base_fcn_r18_1000e_adamw_synapse import * # noqa
    # import teacher model and student model
    from ...unet.unet_base_sgd_synapse import model as teacher_model  # noqa
    from ...unet.unet_tiny_sgd_synapse import model as student_model  # noqa

teacher_ckpt = 'ckpts/unet50_sgd_synapse/best_Dice_83-29_epoch_780.pth'  # noqa: E501
model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=ConfigurableDistiller,
        distill_losses=dict(
            loss_cwd=dict(type=ChannelWiseDivergence, tau=4, loss_weight=1)),
        student_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='segmentor')),
        teacher_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='segmentor')),
        loss_forward_mappings=dict(
            loss_cwd=dict(
                preds_S=dict(from_student=True, recorder='logits'),
                preds_T=dict(from_student=False, recorder='logits')))))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='cwd_unet_b_unet_s-1000e-adamw'),
        )
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
