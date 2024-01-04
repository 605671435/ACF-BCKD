# Copyright (c) OpenMMLab. All rights reserved.

from .fcn_head import FCNHead
from .unet_head import UNetHead
from .ham_head import HamHead
from .dsn_head import DSNHead, MSDSNHead
from .ea_head import EAHead

__all__ = [
    'FCNHead', 'UNetHead', 'HamHead', 'DSNHead', 'EAHead', 'MSDSNHead'
]
