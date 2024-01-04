# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import time
import copy
import os
import os.path as osp
from typing import Optional

from mmengine.config import Config, DictAction
from mmengine.fileio import load
from seg.utils.train_single_fold import train_single_run, test_single_run
from mmengine.logging import print_log
from seg.utils import register_all_modules
import wandb


def find_latest_run(path: str) -> Optional[str]:
    """Find the latest run from the given path.

    Refer to https://github.com/facebookresearch/fvcore/blob/main/fvcore/common/checkpoint.py  # noqa: E501

    Args:
        path(str): The path to find run.

    Returns:
        str or None: File path of the latest run.
    """
    save_file = osp.join(path, 'last_run')
    last_saved: Optional[str]
    if os.path.exists(save_file):
        with open(save_file) as f:
            last_saved = f.read().strip()
    else:
        print_log('Did not find last_run to be resumed.', level=logging.WARNING)
        last_saved = None
    return last_saved

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--num-runs',
        default=3,
        type=int,
        help='The number of all runs.')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--test',
        action='store_true',
        default=False,
        help='whether test when training')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
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
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

if __name__ == '__main__':
    # main()
    args = parse_args()

    # load config
    # register all modules in mmseg into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules(init_default_scope=False)
    os.environ['WANDB_MODE'] = 'offline'
    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume from the previous experiment
    if args.resume:
        assert osp.isdir(args.work_dir)
        experiment_name = osp.basename(args.work_dir)
        cfg.work_dir = args.work_dir
        last_run = find_latest_run(args.work_dir)
        # last_run = os.listdir(args.work_dir)[-1][-1]
        resume_run = int(last_run[-1])
        print_log(f'Resume from last run {resume_run}', logger='current', level=logging.INFO)
    else:
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        experiment_name = f'{args.num_runs}-run_{timestamp}'
        cfg.work_dir = osp.join(cfg.work_dir,
                                experiment_name)
        resume_run = 0

    runs = range(resume_run, args.num_runs)
    resume = args.resume
    for run in runs:
        os.environ['WANDB_MODE'] = 'offline'
        # cfg.train_cfg.max_iters = 50
        # cfg.train_cfg.val_interval = 50
        cfg_ = copy.deepcopy(cfg)
        train_single_run(cfg_, args.num_runs, run, experiment_name, resume)
        if args.test:
            cfg_ = copy.deepcopy(cfg)
            test_single_run(cfg_, args.num_runs, run, experiment_name)
        resume = False
        if wandb.run is not None:
            wandb.join()

    if not args.test:
        print_log(
            'Multi runs have finished, you can test all of your runs by running:',
            logger='current',
            level=logging.INFO)
        print_log(
            f'python multi-test.py {args.config} {cfg.work_dir}',
            logger='current',
            level=logging.INFO)



