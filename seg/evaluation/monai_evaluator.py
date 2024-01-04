from mmengine.evaluator import Evaluator
from typing import Any, Iterator, List, Optional, Sequence, Union

from mmengine.dataset import pseudo_collate
from mmengine.registry import EVALUATOR, METRICS
from mmengine.structures import BaseDataElement


class MonaiEvaluator(Evaluator):
    def process(self,
                data_samples: Sequence[BaseDataElement],
                data_batch: Optional[Any] = None):
        """Convert ``BaseDataSample`` to dict and invoke process method of each
        metric.

        Args:
            data_samples (Sequence[BaseDataElement]): predictions of the model,
                and the ground truth of the validation set.
            data_batch (Any, optional): A batch of data from the dataloader.
        """

        for metric in self.metrics:
            metric.process(data_batch, data_samples)