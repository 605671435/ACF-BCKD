# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import time
import copy
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.fileio import load
from seg.utils.train_single_fold import test_single_run, RUN_INFO_FILE
from mmengine.logging import print_log
from seg.utils import register_all_modules
import wandb

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('work_dir', help='checkpoint file')
    parser.add_argument(
        '--resume-run',
        type=int,
        default=0,
        help='resume from the latest checkpoint in the work_dir automatically')
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

    # re-login wandb if cfg.wandb_keys exists
    # if cfg.wandb_keys:
    #     if osp.exists(cfg.wandb_keys.dir):
    #         import json
    #         with open(cfg.wandb_keys.dir) as fp:
    #             os.environ["WANDB_API_KEY"] = json.loads(fp.read())[cfg.wandb_keys.name]

    # resume from the previous experiment
    assert osp.isdir(args.work_dir)
    experiment_name = osp.basename(args.work_dir)
    cfg.work_dir = args.work_dir
    run_list = [file
                for file in os.listdir(args.work_dir) if file.startswith('run') and not file.startswith('run_exp')]
    num_runs = len(run_list)
    runs = range(args.resume_run, num_runs)
    # remove wandb when test
    for i, vis_backends in enumerate(cfg.visualizer.vis_backends):
        if vis_backends['type'] == 'WandbVisBackend':
            cfg.visualizer.vis_backends.pop(i)

    for run in runs:
        os.environ['WANDB_MODE'] = 'offline'
        cfg_ = copy.deepcopy(cfg)
        test_single_run(cfg_, num_runs, run, experiment_name)
        if wandb.run is not None:
            wandb.join()