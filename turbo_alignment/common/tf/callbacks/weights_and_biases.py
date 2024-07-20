from typing import Any

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from wandb.sdk.lib.disabled import RunDisabled
from wandb.sdk.wandb_run import Run

from turbo_alignment.common.logging import get_project_logger

logger = get_project_logger()


class BaseWandbCallback(TrainerCallback):
    def __init__(self, wandb_run: Run | RunDisabled) -> None:
        super().__init__()
        self._wandb_run = wandb_run

    def on_log(
        self,
        _args: TrainingArguments,
        state: TrainerState,
        _control: TrainerControl,
        **kwargs,
    ) -> None:
        logs = kwargs.get('logs', {})
        self._log(logs=logs, state=state)

    def _log(self, logs: dict[str, Any], state: TrainerState) -> None:
        rewritten_logs = self._rewrite_logs(logs)
        self._wandb_run.log({**rewritten_logs, 'train/global_step': state.global_step}, step=state.global_step)

    @staticmethod
    def _rewrite_logs(logs: dict[str, Any]) -> dict[str, Any]:
        rewritten_logs = {}
        eval_prefix = 'eval_'
        eval_prefix_len = len(eval_prefix)
        test_prefix = 'test_'
        test_prefix_len = len(test_prefix)
        train_prefix = 'train_'
        train_prefix_len = len(train_prefix)
        cherry_pick_prefix = 'cherry_pick_'
        cherry_pick_prefix_len = len(cherry_pick_prefix)
        for k, v in logs.items():
            if k.startswith(eval_prefix):
                rewritten_logs['eval/' + k[eval_prefix_len:]] = v
            elif k.startswith(test_prefix):
                rewritten_logs['test/' + k[test_prefix_len:]] = v
            elif k.startswith(train_prefix):
                rewritten_logs['train/' + k[train_prefix_len:]] = v
            elif k.startswith(cherry_pick_prefix):
                rewritten_logs['cherry_pick/' + k[cherry_pick_prefix_len:]] = v
            else:
                rewritten_logs['train/' + k] = v

        return rewritten_logs
