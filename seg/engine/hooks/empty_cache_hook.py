# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

from mmengine.hooks import EmptyCacheHook as _EmptyCacheHook
from seg.datasets.monai_dataset import CacheMonaiDataset
DATA_BATCH = Optional[Union[dict, tuple, list]]


class EmptyCacheHook(_EmptyCacheHook):

    def after_run(self, runner) -> None:
        dataset: CacheMonaiDataset = getattr(runner.train_dataloader, 'dataset')
        if isinstance(dataset, CacheMonaiDataset):
            dataset.clear_cache()
