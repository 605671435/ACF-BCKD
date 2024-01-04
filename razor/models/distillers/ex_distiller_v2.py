# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

from mmengine.model import BaseModel
from mmengine.registry import MODELS
from torch import nn

from .configurable_distiller import ConfigurableDistiller


class EXDistiller(ConfigurableDistiller):
    def __init__(self,
                 student_plugins: Dict,
                 **kwargs):
        super().__init__(**kwargs)
        self.student_plugins = student_plugins

    # def prepare_from_student(self, model: BaseModel) -> None:
    #     """Initialize student recorders."""
    #
    #     for key, recorder in self.student_recorders.recorders.items():
    #         founded = False
    #         for name, module in model.named_modules():
    #             if name == recorder.source:
    #                 if key in self.student_plugins.keys():
    #                     plugin = MODELS.build(self.student_plugins[key])
    #                     # new_module = nn.Sequential(module, plugin)
    #                     # module = new_module
    #                     # new_module = nn.Sequential(module, plugin)
    #                     model.get_submodule(name) = nn.Sequential(module, plugin)
    #                     # module = model.get_submodule(name)[0]
    #                     module = module.get_submodule('1.attn')
    #                     module.register_forward_hook(recorder.forward_hook)
    #                 else:
    #                     module.register_forward_hook(recorder.forward_hook)
    #                 founded = True
    #                 break
    #         recorder._initialized = True
    #
    #         assert founded, f'"{recorder.source}" is not in the model.'
