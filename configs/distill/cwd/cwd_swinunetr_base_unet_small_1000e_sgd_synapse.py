from mmengine.config import read_base

with read_base():
    from .cwd_swinunetr_base_fcn_r18_1000e_sgd_synapse import * # noqa
    # import teacher model and student model
    from ...swin_unetr.swinunetr_base_5000e_synapse import model as teacher_model  # noqa
    from ...unet.unet_small_sgd_synapse import model as student_model  # noqa

model['architecture'] = dict(cfg_path=student_model, pretrained=False)
find_unused_parameters = True

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='cwd_swinunetr_b_unet_r50-1000e'),
        )
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
