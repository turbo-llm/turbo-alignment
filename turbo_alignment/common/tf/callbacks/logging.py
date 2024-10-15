from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from clearml import Task
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from wandb.sdk.lib.disabled import RunDisabled
from wandb.sdk.wandb_run import Run

import wandb
from turbo_alignment.common.logging import get_project_logger

logger = get_project_logger()


class LoggingCallback(TrainerCallback, ABC):
    def on_log(
        self,
        _args: TrainingArguments,
        state: TrainerState,
        _control: TrainerControl,
        **kwargs,
    ) -> None:
        logs = kwargs.get('logs', {})
        self._log(logs=logs, state=state)

    @abstractmethod
    def _log(self, logs: dict[str, Any], state: TrainerState) -> None:
        ...

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
            if isinstance(v, pd.DataFrame):
                v = wandb.Table(dataframe=v)

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


class WandbLoggingCallback(LoggingCallback):
    def __init__(self, wandb_run: Run | RunDisabled) -> None:
        super().__init__()
        self._wandb_run = wandb_run

    def _log(self, logs: dict[str, Any], state: TrainerState) -> None:
        rewritten_logs: dict[str, Any] = self._rewrite_logs(logs)
        self._wandb_run.log({**rewritten_logs, 'train/global_step': state.global_step}, step=state.global_step)


class ClearMLLoggingCallback(LoggingCallback):
    def __init__(self, clearml_task: Task) -> None:
        super().__init__()
        self._clearml_task = clearml_task

    def _log(self, logs: dict[str, Any], state: TrainerState) -> None:
        rewritten_logs: dict[str, Any] = self._rewrite_logs(logs)

        single_value_scalars: list[str] = [
            'train_runtime',
            'train_samples_per_second',
            'train_steps_per_second',
            'train_loss',
            'epoch',
        ]

        for k, v in rewritten_logs.items():
            title, series = k.split('/')[0], '/'.join(k.split('/')[1:])

            if isinstance(v, (int, float, np.floating, np.integer)):
                if k in single_value_scalars:
                    self._clearml_task.get_logger().report_single_value(name=k, value=v)
                else:
                    self._clearml_task.get_logger().report_scalar(
                        title=title,
                        series=series,
                        value=v,
                        iteration=state.global_step,
                    )
            elif isinstance(v, pd.DataFrame):
                self._clearml_task.get_logger().report_table(
                    title=title,
                    series=series,
                    table_plot=v,
                    iteration=state.global_step,
                )
            else:
                logger.warning(
                    'Trainer is attempting to log a value of '
                    f'"{v}" of type {type(v)} for key "{k}". '
                    "This invocation of ClearML logger's function "
                    'is incorrect so this attribute was dropped. '
                )
