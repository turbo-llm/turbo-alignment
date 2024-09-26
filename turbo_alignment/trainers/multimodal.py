import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.constants import DISABLE_LOSS_LABEL
from turbo_alignment.trainers.multigpu import MultiGPUCherryPicksTrainer

logger = get_project_logger()


class MultimodalTrainer(MultiGPUCherryPicksTrainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        logger.info('Running custom _save_checkpoint')
        checkpoint_folder = f'{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}'
        run_dir = self._get_output_dir(trial=trial)
        output_dir = Path(os.path.join(run_dir, checkpoint_folder))

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

        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

        (output_dir / 'projections').mkdir(parents=True, exist_ok=True)
        (output_dir / 'adapter').mkdir(parents=True, exist_ok=True)
        (output_dir / 'tokenizer').mkdir(parents=True, exist_ok=True)

        if isinstance(model, torch.nn.DataParallel):
            torch.save(
                model.module.modality_adapters.state_dict(), output_dir / 'projections' / 'modality_adapters.pt'
            )
            model.module.language_model.save_pretrained(output_dir / 'adapter')
        else:
            torch.save(model.modality_adapters.state_dict(), output_dir / 'projections' / 'modality_adapters.pt')
            model.language_model.save_pretrained(output_dir / 'adapter')

        self.tokenizer.save_pretrained(output_dir / 'tokenizer')


@dataclass
class MultimodalTrainingArguments(TrainingArguments):
    average_log_prob: bool = False
    gamma: float = 0.0001
