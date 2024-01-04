from seg.engine.runner.monai_runner import MonaiRunner
from mmengine.visualization.vis_backend import LocalVisBackend, WandbVisBackend
from mmseg.visualization.local_visualizer import SegLocalVisualizer

runner_type = MonaiRunner
default_scope = None
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='mmsegmentation', name='exp'),
        define_metric_cfg=dict(Dice='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
test_mode = False
save = False
