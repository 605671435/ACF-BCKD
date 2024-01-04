# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import tempfile
from pathlib import Path

import torch
from mmengine import Config, DictAction
from mmengine.utils import is_tuple_of
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.analysis.complexity_analysis import parameter_count, flop_count, FlopAnalyzer

from mmseg.models import BaseSegmentor
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample

from seg.configs_mapping import config_mapping

try:
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0 to use this script.')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get the FLOPs of a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[512, 512],
        help='input image size')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def inference(args: argparse.Namespace, logger: MMLogger, module: str) -> tuple:
    config_name = Path(args.config)

    if config_name.name.startswith('fcn_r18') \
            or config_name.name.startswith('fcn_fcanet18'):
        # fcn_r18
        base = dict({
            'flops': 49405755392,
            'params': 11195168})
    if config_name.name.startswith('fcn_r50') \
            or config_name.name.startswith('fcn_fcanet50'):
        # fcn_r50
        base = dict({
            'flops': 101155078144,
            'params': 23526688})
    if config_name.name.startswith('unet_r18') \
            or config_name.name.startswith('unet_fcanet18'):
        # unet_r18
        base = dict({
            'flops': 9115795456,
            'params': 11160640})
    if config_name.name.startswith('unet_r50') \
            or config_name.name.startswith('unet_fcanet50'):
        # unet_r50
        base = dict({
            'flops': 21174419456,
            'params': 23448640})
    if not config_name.exists():
        logger.error(f'Config file {config_name} does not exist')

    cfg: Config = Config.fromfile(config_name)
    cfg.work_dir = tempfile.TemporaryDirectory().name
    cfg.log_level = 'WARN'
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('scope', 'seg'))

    if len(args.shape) == 1:
        input_shape = (1, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')
    result = {}

    model: BaseSegmentor = MODELS.build(cfg.model)
    if hasattr(model, 'auxiliary_head'):
        model.auxiliary_head = None
    if torch.cuda.is_available():
        model.cuda()
    model = revert_sync_batchnorm(model)
    result['ori_shape'] = input_shape[-2:]
    result['pad_shape'] = input_shape[-2:]
    data_batch = {
        'inputs': [torch.rand(input_shape)],
        'data_samples': [SegDataSample(metainfo=result)]
    }
    data = model.data_preprocessor(data_batch)
    model.eval()
    if cfg.model.decode_head.type in ['MaskFormerHead', 'Mask2FormerHead']:
        # TODO: Support MaskFormer and Mask2Former
        raise NotImplementedError('MaskFormer and Mask2Former are not '
                                  'supported yet.')
    # outputs = get_model_complexity_info(
    #     model,
    #     input_shape,
    #     # inputs=data['inputs'],
    #     show_table=True,
    #     show_arch=True)
    device = next(model.parameters()).device
    if is_tuple_of(input_shape, int):  # tuple of int, construct one tensor
        inputs = (torch.randn(1, *input_shape).to(device),)
    elif is_tuple_of(input_shape, tuple) and all([
        is_tuple_of(one_input_shape, int)
        for one_input_shape in input_shape  # type: ignore
    ]):  # tuple of tuple of int, construct multiple tensors
        inputs = tuple([
            torch.randn(1, *one_input_shape).to(device)
            for one_input_shape in input_shape  # type: ignore
        ])
    else:
        raise ValueError(
            '"input_shape" should be either a `tuple of int` (to construct'
            'one input tensor) or a `tuple of tuple of int` (to construct'
            'multiple input tensors).')
    flop_handler = FlopAnalyzer(model, inputs)
    flops = flop_handler.by_module()

    params = dict(parameter_count(model))

    s_params = 0
    s_flops = 0
    for source in params.keys():
        if source.endswith(module):
            s_params += params[source]
            s_flops += flops[source]
    return _format_size(s_params), _format_size(s_flops)

def my_format_size(x: int,
                 sig_figs: int = 3,
                 hide_zero: bool = False,
                 f: str = 'G') -> str:
    """Formats an integer for printing in a table or model representation.

    Expresses the number in terms of 'kilo', 'mega', etc., using
    'K', 'M', etc. as a suffix.

    Args:
        x (int): The integer to format.
        sig_figs (int): The number of significant figures to keep.
            Defaults to 3.
        hide_zero (bool): If True, x=0 is replaced with an empty string
            instead of '0'. Defaults to False.

    Returns:
        str: The formatted string.
    """
    if hide_zero and x == 0:
        return ''

    def fmt(x: float) -> str:
        # use fixed point to avoid scientific notation
        return f'{{:.{sig_figs}f}}'.format(x).rstrip('0').rstrip('.')

    if f == 'G':
        return fmt(x / 1e9) + 'G'
    if f == 'M':
        return fmt(x / 1e6) + 'M'
    return str(x)

def main():

    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')
    module = 'gau'
    result = inference(args, logger, module)
    print(f'results of {module} is: params: {result[0]}, flops: {result[1]}')
if __name__ == '__main__':
    main()