# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
import sys

import numpy as np
import torch
import gc
from mmengine.fileio import dump, load
from mmengine.utils.path import mkdir_or_exist
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.registry import RUNNERS
RUN_INFO_FILE = 'run_exp.json'
TEST_INFO_FILE = 'test_exp.json'


def train_single_run(cfg, num_runs, run, experiment_name, resume=False):
    root_dir = cfg.work_dir
    cfg.work_dir = osp.join(root_dir, f'run{run}')
    cfg.resume = resume
    mkdir_or_exist(cfg.work_dir)
    save_file = osp.join(osp.abspath(root_dir), 'last_run')
    RUN_INFO_FILE = osp.splitext(osp.basename(cfg.filename))[0] + '.json'
    with open(save_file, 'w') as f:
        f.write(osp.abspath(cfg.work_dir))  # type: ignore

    # change tags of wandb
    for i, vis_backends in enumerate(cfg.visualizer.vis_backends):
        if vis_backends['type'] == 'WandbVisBackend':
            cfg.visualizer.vis_backends[i]['init_kwargs'].setdefault('tags',
                                                                     [f'run{run}', experiment_name])
            cfg.visualizer.vis_backends[i]['init_kwargs'].setdefault('reinit', True)
            break

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)
    runner.logger.info(
        f'----------- multi-train: [{run + 1}/{num_runs}] ----------- ')

    class SaveInfoHook(Hook):
        def after_train(self, runner):
            best_metrics = runner.message_hub.get_info('best_metrics')
            for key, value in best_metrics.items():
                if np.isnan(value):
                    best_metrics[key] = 0
            # runner.visualizer.add_scalars(
            #     {f'best/{k}': v for k, v in best_metrics.items() if k.find('(') == -1})
            ex_dir = osp.join(root_dir, RUN_INFO_FILE)
            if osp.isfile(ex_dir):
                info = load(ex_dir)
                info['run_info'].setdefault(f'run{run}', best_metrics)
                # average_info = info['average_info']
                for key in info['average_info'].keys():
                    metrics = [v[key] for v in info['run_info'].values()]
                    # metrics = np.array(metrics)
                    average_metrics = np.round(np.nanmean(metrics), 2)
                    std_metrics = np.round(np.nanstd(metrics), 2)
                    info['simple_info'][key] = str(f'{average_metrics}±{std_metrics}')
                    info['average_info'][key] = str(average_metrics)
                    info['std_info'][key] = str(std_metrics)
                dump(info, osp.join(root_dir, RUN_INFO_FILE))
            else:
                run_info = dict({f'run{run}': best_metrics})
                average_info = best_metrics
                info = dict(run_info=run_info,
                            simple_info=average_info,
                            average_info=average_info,
                            std_info=average_info)
                dump(info, osp.join(root_dir, RUN_INFO_FILE))
            runner.visualizer.add_scalars(
                {f'average/{k}': v for k, v in info['simple_info'].items() if k.find('(') == -1})

    runner.register_hook(SaveInfoHook(), 'LOWEST')

    # start training
    runner.train()

    runner.visualizer._instance_dict.clear()
    # del runner
    return

def test_single_run(cfg, num_runs, run, experiment_name):
    root_dir = cfg.work_dir
    cfg.work_dir = osp.join(root_dir, f'run{run}')
    for file in os.listdir(cfg.work_dir):
        if file.startswith('best'):
            cfg.load_from = osp.join(cfg.work_dir, file)
            break
    assert cfg.load_from is not None
    TEST_INFO_FILE = osp.splitext(osp.basename(cfg.filename))[0] + '_test.json'
    # change tags of wandb
    for i, vis_backends in enumerate(cfg.visualizer.vis_backends):
        if vis_backends['type'] == 'WandbVisBackend':
            cfg.visualizer.vis_backends[i]['init_kwargs'].setdefault('tags',
                                                                     [f'run{run}', experiment_name])
            cfg.visualizer.vis_backends[i]['init_kwargs'].setdefault('reinit', True)
            break

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)
    runner.logger.info(
        f'----------- multi-test: [{run + 1}/{num_runs}] ----------- ')

    class SaveInfoHook(Hook):
        def after_test_epoch(self, runner, metrics):
            for key, value in metrics.items():
                if np.isnan(value):
                    metrics[key] = 0
            best_metrics = metrics
            ex_dir = osp.join(root_dir, TEST_INFO_FILE)
            if osp.isfile(ex_dir):
                info = load(ex_dir)
                info['run_info'].setdefault(f'run{run}', best_metrics)
                # average_info = info['average_info']
                for key in info['average_info'].keys():
                    metrics = [v[key] for v in info['run_info'].values()]
                    # metrics = np.array(metrics)
                    average_metrics = np.round(np.nanmean(metrics), 2)
                    std_metrics = np.round(np.nanstd(metrics), 2)
                    info['simple_info'][key] = str(f'{average_metrics}±{std_metrics}')
                    info['average_info'][key] = str(average_metrics)
                    info['std_info'][key] = str(std_metrics)
                dump(info, osp.join(root_dir, TEST_INFO_FILE))
            else:
                run_info = dict({f'run{run}': best_metrics})
                average_info = best_metrics
                info = dict(run_info=run_info,
                            simple_info=average_info,
                            average_info=average_info,
                            std_info=average_info)
                dump(info, osp.join(root_dir, TEST_INFO_FILE))
            runner.visualizer.add_scalars({f'average/{k}': v for k, v in info['average_info'].items()})
    runner.register_hook(SaveInfoHook(), 'LOWEST')

    # start training
    runner.test()

    # del runner
    runner.visualizer._instance_dict.clear()
    return
