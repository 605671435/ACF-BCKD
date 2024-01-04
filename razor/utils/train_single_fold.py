# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
import sys

import numpy as np
import torch
import gc
from mmengine.fileio import dump, load
from mmengine.hooks import Hook
from mmengine.runner import Runner
FOLD_INFO_FILE = 'kfold_exp.json'
RUN_INFO_FILE = 'run_exp.json'
TEST_INFO_FILE = 'test_exp.json'
def train_single_fold(cfg, num_splits, fold, experiment_name, resume=False):
    root_dir = cfg.work_dir
    cfg.work_dir = osp.join(root_dir, f'fold{fold}')

    dataset = cfg.train_dataloader.dataset

    # wrap the dataset cfg
    def wrap_dataset(dataset, test_mode):
        return dict(
            type='KFoldDataset',
            dataset=dataset,
            fold=fold,
            num_splits=num_splits,
            seed=cfg.kfold_split_seed,
            test_mode=test_mode,
        )

    train_dataset = copy.deepcopy(dataset)
    cfg.train_dataloader.dataset = wrap_dataset(train_dataset, False)

    if cfg.val_dataloader is not None:
        if 'pipeline' not in cfg.val_dataloader.dataset:
            raise ValueError(
                'Cannot find `pipeline` in the validation dataset. '
                "If you are using dataset wrapper, please don't use this "
                'tool to act kfold cross validation. '
                'Please write config files manually.')
        val_dataset = copy.deepcopy(dataset)
        val_dataset['pipeline'] = cfg.val_dataloader.dataset.pipeline
        cfg.val_dataloader.dataset = wrap_dataset(val_dataset, True)
    if cfg.test_dataloader is not None:
        if 'pipeline' not in cfg.test_dataloader.dataset:
            raise ValueError(
                'Cannot find `pipeline` in the test dataset. '
                "If you are using dataset wrapper, please don't use this "
                'tool to act kfold cross validation. '
                'Please write config files manually.')
        test_dataset = copy.deepcopy(dataset)
        test_dataset['pipeline'] = cfg.test_dataloader.dataset.pipeline
        cfg.test_dataloader.dataset = wrap_dataset(test_dataset, True)

    # change tags of wandb
    for i, vis_backends in enumerate(cfg.visualizer.vis_backends):
        if vis_backends['type'] == 'WandbVisBackend':
            cfg.visualizer.vis_backends[i]['init_kwargs'].setdefault('tags',
                                                                     [f'fold{fold}', experiment_name])
            cfg.visualizer.vis_backends[i]['init_kwargs'].setdefault('reinit', True)
            break

    # from mmengine.registry import VISUALIZERS
    # VISUALIZERS.module_dict['Visualizer']._instance_dict.clear()
    # build the runner from config
    runner = Runner.from_cfg(cfg)
    runner.logger.info(
        f'----------- Cross-validation: [{fold+1}/{num_splits}] ----------- ')
    # runner.logger.info(f'Train dataset: \n{runner.train_dataloader.dataset}')
    # runner.logger.info(f'Validation dataset: \n{runner.val_dataloader.dataset}')

    class SaveInfoHook(Hook):
        def after_train_epoch(self, runner):
            best_metrics = runner.message_hub.get_info('best_metrics')
            runner.visualizer.add_scalars({f'best/{k}': v for k, v in best_metrics.items()})
            ex_dir = osp.join(root_dir, FOLD_INFO_FILE)
            if osp.isfile(ex_dir):
                info = load(ex_dir)
                info['fold_info'].setdefault(f'fold{fold}', dict(best_metrics=best_metrics))
                # average_info = info['average_info']
                for key in info['average_info'].keys():
                    metrics = [v['best_metrics'][key] for v in info['fold_info'].values()]
                    # metrics = np.array(metrics)
                    average_metrics = np.round(np.nanmean(metrics), 2)
                    info['average_info'][key] = average_metrics
                dump(info, osp.join(root_dir, FOLD_INFO_FILE))
            else:
                exp_info = dict(kfold_split_seed=str(cfg.kfold_split_seed))
                fold_info = dict({f'fold{fold}': dict(best_metrics=best_metrics)})
                average_info = best_metrics
                info = dict(exp_info=exp_info,
                            fold_info=fold_info,
                            average_info=average_info)
                dump(info, osp.join(root_dir, FOLD_INFO_FILE))
            runner.visualizer.add_scalars({f'average/{k}': v for k, v in info['average_info'].items()})

    runner.register_hook(SaveInfoHook(), 'LOWEST')

    # start training
    runner.train()
    # runner.visualizer.close()
    runner.visualizer._instance_dict.clear()
    return
    # sys.exit()

    # process environment to avoid error when train twice more
    # gc.collect()
    # torch.cuda.empty_cache()
    # if hasattr(runner, 'train_loop'):
    #     runner.train_loop.dataloader_iterator._iterator.__del__()
    # runner.visualizer.close()

def train_single_run(cfg, num_runs, run, experiment_name, resume=False):
    root_dir = cfg.work_dir
    cfg.work_dir = osp.join(root_dir, f'run{run}')
    cfg.resume = resume
    # change tags of wandb
    for i, vis_backends in enumerate(cfg.visualizer.vis_backends):
        if vis_backends['type'] == 'WandbVisBackend':
            cfg.visualizer.vis_backends[i]['init_kwargs'].setdefault('tags',
                                                                     [f'run{run}', experiment_name])
            cfg.visualizer.vis_backends[i]['init_kwargs'].setdefault('reinit', True)
            break

    runner = Runner.from_cfg(cfg)
    runner.logger.info(
        f'----------- multi-train: [{run + 1}/{num_runs}] ----------- ')

    class SaveInfoHook(Hook):
        def after_train_epoch(self, runner):
            best_metrics = runner.message_hub.get_info('best_metrics')
            runner.visualizer.add_scalars({f'best/{k}': v for k, v in best_metrics.items()})
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
                    info['average_info'][key] = str(f'{average_metrics}±{std_metrics}')
                dump(info, osp.join(root_dir, RUN_INFO_FILE))
            else:
                run_info = dict({f'run{run}': best_metrics})
                average_info = best_metrics
                info = dict(run_info=run_info,
                            average_info=average_info)
                dump(info, osp.join(root_dir, RUN_INFO_FILE))
            runner.visualizer.add_scalars({f'average/{k}': v for k, v in info['average_info'].items()})

    runner.register_hook(SaveInfoHook(), 'LOWEST')

    # start training
    runner.train()

    runner.visualizer._instance_dict.clear()
    return

def test_single_run(cfg, num_runs, run, experiment_name):
    root_dir = cfg.work_dir
    cfg.work_dir = osp.join(root_dir, f'run{run}')
    for file in os.listdir(cfg.work_dir):
        if file.startswith('best'):
            cfg.load_from = osp.join(cfg.work_dir, file)
            break
    assert cfg.load_from is not None

    runner = Runner.from_cfg(cfg)
    runner.logger.info(
        f'----------- multi-test: [{run + 1}/{num_runs}] ----------- ')

    class SaveInfoHook(Hook):
        def after_test_epoch(self, runner, metrics):
            for key, value in metrics.items():
                np.isnan(value)
                if np.isnan(value):
                    metrics.pop(key, None)
                    break
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
                    info['average_info'][key] = str(f'{average_metrics}±{std_metrics}')
                dump(info, osp.join(root_dir, TEST_INFO_FILE))
            else:
                run_info = dict({f'run{run}': best_metrics})
                average_info = best_metrics
                info = dict(run_info=run_info,
                            average_info=average_info)
                dump(info, osp.join(root_dir, TEST_INFO_FILE))

    runner.register_hook(SaveInfoHook(), 'LOWEST')

    # start training
    runner.test()

    return
