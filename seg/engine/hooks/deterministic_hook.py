import torch
import os
from seg.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.utils import digit_version
@HOOKS.register_module()
class DeterministicHook(Hook):
    def __init__(self,
                 deterministic=True,
                 warn_only=False,
                 cublas_cfg=':4096:8',
                 **kwargs):
        super(DeterministicHook, self).__init__(**kwargs)
        self.deterministic = deterministic
        self.warn_only = warn_only
        allowed_cublas_cfg = [':4096:8', ':16:8']
        assert cublas_cfg in allowed_cublas_cfg, 'os_cfg should be set to :4096:8 or :16:8.'
        self.cublas_cfg = cublas_cfg

    def before_run(self, runner) -> None:
        if self.deterministic:
            torch.backends.cudnn.benchmark = False
            # avoid error when cuda>=10.2:
            if digit_version(torch.version.cuda) >= digit_version('10.2'):
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = self.cublas_cfg
            # is deterministic for convolution operators:
            torch.backends.cudnn.deterministic = True
            # whether to use deterministic algorithms.
            # Note that non-deterministic algorithms such as up-sampling
            # will throw an error when warn_only is False.
            torch.use_deterministic_algorithms(True, warn_only=self.warn_only)

