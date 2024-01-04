import argparse

from seg.apis import init_model
from mmseg.utils import register_all_modules
from mmengine.registry import MODELS
from mmengine import Config
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
    # show all named module in the model and use it in source list below
    print(model)
    for name, module in model.named_modules():
        print(name)
if __name__ == '__main__':
    main()
