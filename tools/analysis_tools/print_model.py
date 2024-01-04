import argparse

from seg.apis import init_model
from mmengine.registry import MODELS
from mmengine.config import Config
from mmseg.utils import register_all_modules
from seg.models.utils.ex_kd import EX_KD_3D
from torch.nn import ConvTranspose3d
from monai.networks.blocks.convolutions import ResidualUnit

def parse_args():
    parser = argparse.ArgumentParser(description='Print a model')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    register_all_modules(False)
    config = args.config
    config = Config.fromfile(config)
    model = MODELS.build(config.model)

    print(model)
    resuidual_unit = []
    # show all named module in the model and use it in source list below
    for name, module in model.named_modules():
        if isinstance(module, ConvTranspose3d):
            resuidual_unit.append(name)
        print(name)
    print(resuidual_unit)


if __name__ == '__main__':
    main()
