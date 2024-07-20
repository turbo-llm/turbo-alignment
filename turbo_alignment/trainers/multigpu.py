import os
from typing import Callable

import numpy as np
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    DefaultFlowCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PrinterCallback,
    ProgressCallback,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainingArguments,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from turbo_alignment.common.tf.callbacks.common import WandbMetricsCallbackHandler


class MultiGPUCherryPicksTrainer(Trainer):
    def __init__(
        self,
        model: PreTrainedModel | nn.Module,
        data_collator: Callable,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase | None = None,
        callbacks: list[TrainerCallback] | None = None,
        **kwargs,
    ):
        ref_model = kwargs.pop('ref_model', None)
        metrics_kwargs = kwargs.pop('metrics_kwargs', None)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks,
            **kwargs,
        )

        default_callbacks = [DefaultFlowCallback] + get_reporting_integration_callbacks(self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = WandbMetricsCallbackHandler(
            callbacks,
            model,
            tokenizer,
            None,
            None,
            ref_model=ref_model,
            accelerator=self.accelerator,
            metrics_kwargs=metrics_kwargs,
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else ProgressCallback)
        self.control: TrainerControl = self.callback_handler.on_init_end(self.args, self.state, self.control)

    def _save_checkpoint(self, model, trial, metrics=None):  # pylint: disable=unused-argument
        # FIX: https://github.com/huggingface/transformers/issues/28119
        # https://github.com/huggingface/transformers/pull/29370

        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f'{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}'

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)

        if not self.args.save_only_model:
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(output_dir)
            # Save RNG state
            self._save_rng_state(output_dir)

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith('eval_'):
                metric_to_check = f'eval_{metric_to_check}'
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, 'trainer_state.json'))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            # Solely rely on numerical checkpoint id for rotation.
            # mtime is not reliable especially on some fuse fs in cloud environments.
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)
