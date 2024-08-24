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


class TrainerCustomSave(MultiGPUCherryPicksTrainer):
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


class MultimodalTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel | nn.Module],
        data_collator: Callable,
        args: MultimodalTrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        **kwargs,
    ):
        self.data_collator = data_collator

        self._stored_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self._average_log_prob = args.average_log_prob
        self._gamma = args.gamma

        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            None,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
            **kwargs,
        )

    def concatenated_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Объединяем в один прогон батч с нормальной attention маской и с той, где в modality tokens стоят нолики
        """

        no_modality_attention_mask = torch.logical_and(
            batch['attention_mask'],
            torch.logical_not(batch['modality_tokens_mask']),
        )

        concatenated_batch: Dict[str, torch.Tensor] = {
            'input_ids': batch['input_ids'].repeat(2, 1).to(self.accelerator.device),
            'modality_tokens_mask': batch['modality_tokens_mask'].repeat(2, 1).to(self.accelerator.device),
            'attention_mask': torch.cat(
                [
                    batch['attention_mask'],
                    no_modality_attention_mask,
                ],
                dim=0,
            ).to(self.accelerator.device),
            'labels': batch['input_ids'].repeat(2, 1).to(self.accelerator.device),
            'modality_inputs': batch['modality_inputs'] + batch['modality_inputs'],
        }

        return concatenated_batch

    def multimodal_loss(
        self,
        logits: torch.Tensor,
        logps: torch.Tensor,
        ignore_modality_logps: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = (labels != DISABLE_LOSS_LABEL).sum(1)
        # modality_indifference = seq_len / (torch.abs(logps.sum() - ignore_modality_logps.sum()) + self._gamma)
        modality_indifference = seq_len / (logps.sum() - ignore_modality_logps.sum() + self._gamma)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        ce_loss_fct = CrossEntropyLoss(reduction='none')
        loss = ce_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        sample_mean_ce_loss = loss.view(logits.size(0), -1).mean(dim=1)

        loss = (modality_indifference * sample_mean_ce_loss).mean()
        return loss, modality_indifference

    @staticmethod
    def _get_batch_logps(
        logits: torch.Tensor,
        labels: torch.Tensor,
        average_log_prob: bool = False,
    ) -> torch.Tensor:
        if logits.shape[:-1] != labels.shape:
            raise ValueError('Logits (batch and sequence length dim) and labels must have the same shape.')

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != DISABLE_LOSS_LABEL

        labels[labels == DISABLE_LOSS_LABEL] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        return (per_token_logps * loss_mask).sum(-1)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        concatenated_batch = self.concatenated_inputs(batch)
        all_logits = model(**{k: v for k, v in concatenated_batch.items() if k != 'labels'}).logits.to(torch.float32)

        all_logps = self._get_batch_logps(
            all_logits,
            concatenated_batch['labels'],
            average_log_prob=self._average_log_prob,
        )
        chosen_idxs = batch['input_ids'].shape[0]

        logps = all_logps[:chosen_idxs]
        no_modality_logps = all_logps[chosen_idxs:]

        logits = all_logits[:chosen_idxs]
        no_modality_logits = all_logits[chosen_idxs:]
        return logps, no_modality_logps, logits, no_modality_logits

    def get_batch_metrics(
        self,
        batch: Dict[str, Any],
        train_eval: Literal['train', 'eval'] = 'train',
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        def _compute_cpu_mean(tensor: torch.Tensor) -> float:
            return tensor.detach().cpu().mean().item()

        assert 'modality_tokens_mask' in batch
        assert 'attention_mask' in batch
        assert 'input_ids' in batch
        assert 'labels' in batch

        metrics = {}

        (
            logps,
            ignore_modality_logps,
            logits,
            _,
        ) = self.concatenated_forward(self.model, batch)

        loss, modality_indifference = self.multimodal_loss(
            logits,
            logps,
            ignore_modality_logps,
            batch['labels'],
        )

        prefix = 'eval_' if train_eval == 'eval' else ''
        metrics[f'{prefix}logits/modality_indifference'] = _compute_cpu_mean(modality_indifference)
        metrics[f'{prefix}logits/logps'] = _compute_cpu_mean(logps)
        metrics[f'{prefix}logits/ignore_modality_logps'] = _compute_cpu_mean(ignore_modality_logps)

        return loss, metrics

    def compute_loss(
        self,
        _model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        loss, metrics = self.get_batch_metrics(inputs, train_eval='train')

        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval='train')
        if return_outputs:
            return loss, metrics
        return loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if ignore_keys is None:
            if hasattr(model, 'config'):
                ignore_keys = getattr(model.config, 'keys_to_ignore_at_inference', [])
            else:
                ignore_keys = []

        with torch.inference_mode():
            loss, metrics = self.get_batch_metrics(inputs, train_eval='eval')

        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval='eval')
        if prediction_loss_only:
            return loss.detach(), None, None

        logits_dict = {
            'logits_test/chosen': metrics['logits_test/chosen'],
            'logits_test/rejected': metrics['logits_test/rejected'],
        }
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1)  # type: ignore[call-overload, arg-type]
        labels = torch.zeros(logits.shape[0])

        return loss.detach(), logits, labels

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal['train', 'eval'] = 'train') -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: Dict[str, float]) -> None:
        train_eval = 'train' if 'loss' in logs else 'eval'
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)  # pylint: disable=no-member

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

        (output_dir / 'projections').mkdir(parents=True, exist_ok=True)
        (output_dir / 'language_model').mkdir(parents=True, exist_ok=True)
        (output_dir / 'tokenizer').mkdir(parents=True, exist_ok=True)

        torch.save(model.image_linear_projection.state_dict(), output_dir / 'projections' / 'image_connector.pt')
        torch.save(model.audio_linear_projection.state_dict(), output_dir / 'projections' / 'audio_connector.pt')

        model.language_model.save_pretrained(output_dir / 'language_model')
        self.tokenizer.save_pretrained(output_dir / 'tokenizer')
