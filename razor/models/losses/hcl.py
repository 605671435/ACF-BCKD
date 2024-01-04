# https://github.com/dvlab-research/ReviewKD
import torch
import torch.nn as nn
import torch.nn.functional as F
from razor.registry import MODELS

@MODELS.register_module()
class HCL(nn.Module):
    """hierarchical context loss (HCL) function

    Args:
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def forward(
        self,
        s_feature: torch.Tensor,
        t_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Forward computation.

        Args:
            s_feature (torch.Tensor): The student model feature with
                shape (N, C, H, W) or shape (N, C).
            t_feature (torch.Tensor): The teacher model feature with
                shape (N, C, H, W) or shape (N, C).
        """
        n, c, h, w = s_feature.shape
        loss = F.mse_loss(s_feature, t_feature, reduction='mean')
        cnt = 1.0
        tot = 1.0
        for l in [4, 2, 1]:
            if l >= h:
                continue
            tmpfs = F.adaptive_avg_pool2d(s_feature, (l, l))
            tmpft = F.adaptive_avg_pool2d(t_feature, (l, l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
            tot += cnt
        loss = loss / tot

        return self.loss_weight * loss
