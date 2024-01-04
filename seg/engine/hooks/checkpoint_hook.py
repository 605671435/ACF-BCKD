import os.path as osp
from typing import Optional
from mmengine.hooks import CheckpointHook
from mmengine.dist import is_main_process
from mmengine.logging import print_log
from seg.registry import HOOKS

@HOOKS.register_module()
class MyCheckpointHook(CheckpointHook):
    def __init__(self, **kwargs):
        super(MyCheckpointHook, self).__init__(**kwargs)

    def _save_best_checkpoint(self, runner, metrics) -> None:
        """Save the current checkpoint and delete outdated checkpoint.

        Args:
            runner (Runner): The runner of the training process.
            metrics (dict): Evaluation results of all metrics.
        """
        if not self.save_best:
            return

        if self.by_epoch:
            ckpt_filename = self.filename_tmpl.format(runner.epoch)
            cur_type, cur_time = 'epoch', runner.epoch
        else:
            ckpt_filename = self.filename_tmpl.format(runner.iter)
            cur_type, cur_time = 'iter', runner.iter

        # handle auto in self.key_indicators and self.rules before the loop
        if 'auto' in self.key_indicators:
            self._init_rule(self.rules, [list(metrics.keys())[0]])

        # save best logic
        # get score from messagehub
        for key_indicator, rule in zip(self.key_indicators, self.rules):
            key_score = metrics[key_indicator]

            if len(self.key_indicators) == 1:
                best_score_key = 'best_score'
                runtime_best_ckpt_key = 'best_ckpt'
                best_ckpt_path = self.best_ckpt_path
            else:
                best_score_key = f'best_score_{key_indicator}'
                runtime_best_ckpt_key = f'best_ckpt_{key_indicator}'
                best_ckpt_path = self.best_ckpt_path_dict[key_indicator]

            if best_score_key not in runner.message_hub.runtime_info:
                best_score = self.init_value_map[rule]
            else:
                best_score = runner.message_hub.get_info(best_score_key)

            if key_score is None or not self.is_better_than[key_indicator](
                    key_score, best_score):
                continue

            best_score = key_score

            # update all metrics to message_hub
            best_metrics = dict()
            for key, val in metrics.items():
                best_metrics[f'{key}'] = val
            runner.message_hub.update_info('best_metrics', best_metrics)

            runner.message_hub.update_info(best_score_key, best_score)

            if best_ckpt_path and \
               self.file_client.isfile(best_ckpt_path) and \
               is_main_process():
                self.file_client.remove(best_ckpt_path)
                runner.logger.info(
                    f'The previous best checkpoint {best_ckpt_path} '
                    'is removed')

            best_score_str = f'{best_score:0.2f}'.replace('.', '-')
            best_ckpt_name = f'best_{key_indicator}_{best_score_str}_{ckpt_filename}'
            if len(self.key_indicators) == 1:
                self.best_ckpt_path = self.file_client.join_path(  # type: ignore # noqa: E501
                    self.out_dir, best_ckpt_name)
                runner.message_hub.update_info(runtime_best_ckpt_key,
                                               self.best_ckpt_path)
            else:
                self.best_ckpt_path_dict[
                    key_indicator] = self.file_client.join_path(  # type: ignore # noqa: E501
                        self.out_dir, best_ckpt_name)
                runner.message_hub.update_info(
                    runtime_best_ckpt_key,
                    self.best_ckpt_path_dict[key_indicator])
            runner.save_checkpoint(
                self.out_dir,
                filename=best_ckpt_name,
                file_client_args=self.file_client_args,
                save_optimizer=False,
                save_param_scheduler=False,
                by_epoch=False,
                backend_args=self.backend_args)
            runner.logger.info(
                f'The best checkpoint with {best_score:0.4f} {key_indicator} '
                f'at {cur_time} {cur_type} is saved to {best_ckpt_name}.')

            # save_file = osp.join(runner.work_dir, f'best_checkpoint')
            # filepath = self.file_backend.join_path(self.out_dir, best_ckpt_name)
            # with open(save_file, 'w') as f:
            #     f.write(filepath)

def find_best_checkpoint(path: str) -> Optional[str]:
    """Find the latest checkpoint from the given path.

    Refer to https://github.com/facebookresearch/fvcore/blob/main/fvcore/common/checkpoint.py  # noqa: E501

    Args:
        path(str): The path to find checkpoints.

    Returns:
        str or None: File path of the latest checkpoint.
    """
    save_file = osp.join(path, f'best_checkpoint')
    best_saved: Optional[str]
    if osp.exists(save_file):
        with open(save_file) as f:
            best_saved = f.read().strip()
    else:
        print_log(f'Did not find best_checkpoint to be resumed.')
        best_saved = None
    return best_saved
