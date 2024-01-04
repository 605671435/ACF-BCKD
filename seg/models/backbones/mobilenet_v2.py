# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import build_plugin_layer

from mmseg.models import MobileNetV2
from seg.registry import MODELS
@MODELS.register_module()
class MobileNetV2_EX(MobileNetV2):
    def __init__(self,
                 stage_plugin=dict(type='EX_KD'),
                 out_channels=320,
                 **kwargs):
        super().__init__(**kwargs)
        plugin = stage_plugin.copy()
        layer = getattr(self, self.layers[-1])
        name, plugin_layer = build_plugin_layer(
            plugin,
            in_channels=out_channels,
            postfix=plugin.pop('postfix', ''))
        layer.append(plugin_layer)
