# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from mmseg.models.utils import Encoding
from seg.registry.registry import MODELS

@MODELS.register_module()
class EncModule(nn.Module):
    """Encoding Module used in EncNet.

    Args:
        in_channels (int): Input channels.
        num_codes (int): Number of code words.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self,
                 in_channels,
                 num_codes=32,
                 conv_cfg=None,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.encoding_project = ConvModule(
            in_channels,
            in_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        # TODO: resolve this hack
        # change to 1d
        if norm_cfg is not None:
            encoding_norm_cfg = norm_cfg.copy()
            if encoding_norm_cfg['type'] in ['BN', 'IN']:
                encoding_norm_cfg['type'] += '1d'
            else:
                encoding_norm_cfg['type'] = encoding_norm_cfg['type'].replace(
                    '2d', '1d')
        else:
            # fallback to BN1d
            encoding_norm_cfg = dict(type='BN1d')
        self.encoding = nn.Sequential(
            Encoding(channels=in_channels, num_codes=num_codes),
            build_norm_layer(encoding_norm_cfg, num_codes)[1],
            nn.ReLU(inplace=True))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels), nn.Sigmoid())

    def forward(self, x):
        """Forward function."""
        encoding_projection = self.encoding_project(x)
        encoding_feat = self.encoding(encoding_projection).mean(dim=1)
        batch_size, channels, _, _ = x.size()
        gamma = self.fc(encoding_feat)
        y = gamma.view(batch_size, channels, 1, 1)
        output = F.relu_(x + x * y)
        return output