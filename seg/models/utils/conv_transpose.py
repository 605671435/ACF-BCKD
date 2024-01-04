from torch import nn
from torch.nn import ConvTranspose2d

class InterpConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 with_cp=False,
                 *,
                 norm_cfg=None,
                 act_cfg=None,
                 kernel_size=2,
                 stride=2,
                 padding=0):
        super().__init__()

        self.with_cp = with_cp
        self.interp_upsample = ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)

    def forward(self, x):
        """Forward function."""

        out = self.interp_upsample(x)
        return out
