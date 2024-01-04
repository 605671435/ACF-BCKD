# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Optional

from mmrazor.registry import TASK_UTILS
from mmrazor.models.task_modules.recorder import RecorderManager as _RecorderManager
from mmrazor.models.task_modules.recorder.base_recorder import BaseRecorder

class RecorderManager(_RecorderManager):

    def __init__(self, recorders: Optional[Dict] = None) -> None:

        self._recorders: Dict[str, BaseRecorder] = dict()
        if recorders:
            for name, cfg in recorders.items():
                recorder_cfg = copy.deepcopy(cfg)
                recorder_type = cfg['type']
                if isinstance(recorder_type, str):
                    recorder_type_ = recorder_type + 'Recorder'

                    recorder_cfg['type'] = recorder_type_
                recorder = TASK_UTILS.build(recorder_cfg)

                self._recorders[name] = recorder