# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Optional
from seg.registry import VISBACKENDS
from mmengine.visualization import WandbVisBackend
@VISBACKENDS.register_module()
class WandbVisBackend(WandbVisBackend):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _init_env(self):
        """Setup env for wandb."""
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir, exist_ok=True)  # type: ignore
        if self._init_kwargs is None:
            self._init_kwargs = {'dir': self._save_dir}
        else:
            self._init_kwargs.setdefault('dir', self._save_dir)
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')
        self._init_kwargs.setdefault('settings', wandb.Settings(start_method="fork"))
        wandb.init(**self._init_kwargs)
        self._env_initialized = True
        if self._define_metric_cfg is not None:
            for metric, summary in self._define_metric_cfg.items():
                wandb.define_metric(metric, summary=summary)
        self._wandb = wandb