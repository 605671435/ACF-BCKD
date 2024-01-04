from mmengine.config import read_base

with read_base():
    from .dist_swinunetr_b_fcn_r50_synapse_1000e import *   # noqa
    from ...unet.unet_small_sgd_synapse import model as student_model   # noqa

model['architecture'] = dict(cfg_path=student_model, pretrained=False)


vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='dist-swinunetr-b-unet-small-40k'),
        )
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
