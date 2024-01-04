from .resnet import ResNet
from ..utils.hamburger import Ham


class HamNet(ResNet):
    """ResNetV1c variant described in [1]_.

    Compared with default ResNet(ResNetV1b), ResNetV1c replaces the 7x7 conv in
    the input stem with three 3x3 convs. For more details please refer to `Bag
    of Tricks for Image Classification with Convolutional Neural Networks
    <https://arxiv.org/abs/1812.01187>`_.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        res_layer = getattr(self, self.res_layers[-1])
        out_channel = 512 if self.depth < 50 else 2048
        ham_layer = Ham(in_channels=out_channel)
        res_layer.add_module('ham_layer', ham_layer)
