from mmengine.config import read_base

with read_base():
    from .cwd_swinunetr_base_fcn_r18_1000e_sgd_synapse import * # noqa
    # import teacher model and student model
    from ...swin_unetr.swinunetr_base_5000e_synapse import model as teacher_model  # noqa
    from ...fcn.fcn_r50_sgd_synapse_1000e import model as student_model  # noqa

model.update(
    dict(
        architecture=dict(cfg_path=student_model, pretrained=False),
        teacher=dict(cfg_path=teacher_model, pretrained=False)))
find_unused_parameters = True

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='cwd_swinunetr_b_fcn_r18-80k'),
        )
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
