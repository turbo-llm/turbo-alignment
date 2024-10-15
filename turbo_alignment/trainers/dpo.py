from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch
import torch.nn.functional as F
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

from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.common.tf.callbacks.common import MetricsCallbackHandler
from turbo_alignment.common.tf.callbacks.sync_ref_model import SyncRefModelCallback
from turbo_alignment.constants import DISABLE_LOSS_LABEL
from turbo_alignment.settings.pipelines.train.dpo import (
    APODownLossSettings,
    APOZeroLossSettings,
    CPOLossSettings,
    DPOLossesType,
    ASFTLossSettings,
    HingeLossSettings,
    IPOLossSettings,
    KTOLossSettings,
    ORPOLossSettings,
    SigmoidLossSettings,
    SigmoidLossWithMarginSettings,
    SimPOLossSettings,
    SlicHfLossSettings,
    SyncRefModelSettings,
)
from turbo_alignment.trainers.utils import (
    DPOLossRegistry,
    concatenated_inputs,
    prepare_model,
)

logger = get_project_logger()


@DPOLossRegistry.register(DPOLossesType.SIGMOID)
class SigmoidLoss(DPOLossRegistry):
    def __init__(self, *args, beta: float = 0.1, label_smoothing: float = 0, **kwargs) -> None:
        self.beta = beta
        self.label_smoothing = label_smoothing
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        precomputed_margins: torch.FloatTensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        loss = (
            -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
            - F.logsigmoid(-self.beta * logits) * self.label_smoothing
        )

        return (
            loss,
            chosen_rewards,
            rejected_rewards,
        )


@DPOLossRegistry.register(DPOLossesType.KTO)
class KTOLoss(DPOLossRegistry):
    def __init__(self, *args, beta: float = 0.1, **kwargs) -> None:
        self.beta = beta
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        precomputed_margins: torch.FloatTensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
        rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        loss = torch.cat(
            (
                1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
            ),
            0,
        )

        return loss, chosen_rewards, rejected_rewards


@DPOLossRegistry.register(DPOLossesType.IPO)
class IPOLoss(DPOLossRegistry):
    def __init__(self, *args, beta: float = 0.1, **kwargs) -> None:
        self.beta = beta
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        precomputed_margins: torch.FloatTensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        loss = (logits - 1 / (2 * self.beta)) ** 2

        return loss, chosen_rewards, rejected_rewards


@DPOLossRegistry.register(DPOLossesType.CPO)
class CPOLoss(DPOLossRegistry):
    def __init__(self, *args, beta: float = 0.1, norm: bool, **kwargs) -> None:
        self.beta = beta
        self.norm = norm
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        precomputed_margins: torch.FloatTensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        dpo_loss = -F.logsigmoid(self.beta * logits) if self.norm else -F.logsigmoid(self.beta * pi_logratios)
        sft_loss = -policy_chosen_logps

        loss = dpo_loss + sft_loss

        return loss, chosen_rewards, rejected_rewards


@DPOLossRegistry.register(DPOLossesType.HINGE)
class HingeLoss(DPOLossRegistry):
    def __init__(self, *args, beta: float = 0.1, norm: bool = False, **kwargs) -> None:
        self.beta = beta
        self.norm = norm
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        precomputed_margins: torch.FloatTensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        loss = torch.relu(1 - self.beta * (policy_chosen_logps - policy_rejected_logps))

        if self.norm:
            loss = torch.relu(1 - self.beta * logits)

        return loss, chosen_rewards, rejected_rewards


@DPOLossRegistry.register(DPOLossesType.SLIC_HF)
class SlicHfLoss(DPOLossRegistry):
    def __init__(self, delta: float = 1, beta: float = 1.0, lam: float = 1.0, norm: bool = False) -> None:
        self.delta = delta
        self.beta = beta
        self.norm = norm
        self.lam = lam

    def compute_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        precomputed_margins: torch.FloatTensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        loss = torch.relu(self.delta - self.beta * (policy_chosen_logps - policy_rejected_logps))

        if self.norm:
            loss = torch.relu(self.delta - self.beta * logits)

        if precomputed_margins is not None:
            loss = loss - self.lam * precomputed_margins

        return loss, chosen_rewards, rejected_rewards


@DPOLossRegistry.register(DPOLossesType.SIMPO)
class SimPOLoss(DPOLossRegistry):
    def __init__(self, *args, beta: float = 0.1, gamma: float = 0.1, **kwargs) -> None:
        self.beta = beta
        self.gamma = gamma
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor | None,
        reference_rejected_logps: torch.FloatTensor | None,
        precomputed_margins: torch.FloatTensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps

        logits = pi_logratios - self.gamma

        chosen_rewards = self.beta * (policy_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps).detach()

        loss = -F.logsigmoid(self.beta * logits)

        return (
            loss,
            chosen_rewards,
            rejected_rewards,
        )


@DPOLossRegistry.register(DPOLossesType.ORPO)
class ORPOLoss(DPOLossRegistry):
    def __init__(self, *args, beta: float = 0.1, **kwargs) -> None:
        self.beta = beta
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor | None,
        reference_rejected_logps: torch.FloatTensor | None,
        precomputed_margins: torch.FloatTensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_odds = (policy_chosen_logps - policy_rejected_logps) - (
            torch.log1p(-torch.clamp(torch.exp(policy_chosen_logps), max=1 - 1e-7))
            - torch.log1p(-torch.clamp(torch.exp(policy_rejected_logps), max=1 - 1e-7))
        )

        ratio = -F.logsigmoid(log_odds)
        losses = -policy_chosen_logps + self.beta * ratio

        chosen_rewards = self.beta * policy_chosen_logps.detach()
        rejected_rewards = self.beta * policy_rejected_logps.detach()

        return losses, chosen_rewards, rejected_rewards


@DPOLossRegistry.register(DPOLossesType.ASFT)
class ASFTLoss(DPOLossRegistry):
    def __init__(self, *args, beta: float = 0.1, **kwargs) -> None:
        self.beta = beta
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor | None,
        reference_rejected_logps: torch.FloatTensor | None,
        precomputed_margins: torch.FloatTensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chosen_ratio = policy_chosen_logps - (torch.log1p(-torch.clamp(torch.exp(policy_chosen_logps), max=1 - 1e-7)))
        rejected_ratio = policy_rejected_logps - (
            torch.log1p(-torch.clamp(torch.exp(policy_rejected_logps), max=1 - 1e-7))
        )
        sigm_diff = -F.logsigmoid(chosen_ratio) - F.logsigmoid(-rejected_ratio)

        losses = -policy_chosen_logps + self.beta * sigm_diff

        chosen_rewards = self.beta * policy_chosen_logps.detach()
        rejected_rewards = self.beta * policy_rejected_logps.detach()

        return losses, chosen_rewards, rejected_rewards


@DPOLossRegistry.register(DPOLossesType.SIGMOID_WITH_MARGIN)
class SigmoidLossWithMargin(DPOLossRegistry):
    def __init__(self, *args, beta: float = 0.1, **kwargs) -> None:
        self.beta = beta
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        precomputed_margins: torch.FloatTensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        if precomputed_margins is None:
            raise ValueError('Precomputed margins should not be none when using SigmoidLossWithMargin')

        loss = -F.logsigmoid(self.beta * logits - precomputed_margins)

        return (
            loss,
            chosen_rewards,
            rejected_rewards,
        )


@DPOLossRegistry.register(DPOLossesType.APO_DOWN)
class APODownLoss(DPOLossRegistry):
    def __init__(self, *args, beta: float = 0.1, **kwargs) -> None:
        self.beta = beta
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        precomputed_margins: torch.FloatTensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        losses_chosen = F.sigmoid(self.beta * chosen_logratios)
        losses_rejected = 1 - F.sigmoid(self.beta * (chosen_logratios - rejected_logratios))

        loss = losses_chosen + losses_rejected

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return (
            loss,
            chosen_rewards,
            rejected_rewards,
        )


@DPOLossRegistry.register(DPOLossesType.APO_ZERO)
class APOZeroLoss(DPOLossRegistry):
    def __init__(self, *args, beta: float = 0.1, **kwargs) -> None:
        self.beta = beta
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        precomputed_margins: torch.FloatTensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        losses_chosen = 1 - F.sigmoid(self.beta * chosen_logratios)
        losses_rejected = F.sigmoid(self.beta * rejected_logratios)

        loss = losses_chosen + losses_rejected

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return (
            loss,
            chosen_rewards,
            rejected_rewards,
        )


@dataclass
class DPOTrainingArguments(TrainingArguments):
    loss_settings: (
        SigmoidLossSettings
        | HingeLossSettings
        | IPOLossSettings
        | SlicHfLossSettings
        | KTOLossSettings
        | CPOLossSettings
        | ORPOLossSettings
        | ASFTLossSettings
        | SimPOLossSettings
        | SlicHfLossSettings
        | SigmoidLossWithMarginSettings
        | APOZeroLossSettings
        | APODownLossSettings
    ) = field(
        default_factory=SigmoidLossSettings(loss_type=DPOLossesType.SIGMOID)
    )  # type: ignore[call-overload]
    sync_ref_settings: SyncRefModelSettings = field(  # type: ignore[call-overload]
        default_factory=SyncRefModelSettings()
    )
    use_ref_model: bool = True
    use_sft_model: bool = False
    average_log_prob: bool = False


class DPOTrainer(Trainer):
    """
    Inspired by https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py

    """

    def __init__(
        self,
        model: PreTrainedModel | nn.Module,
        data_collator: Callable,
        args: DPOTrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        ref_model: PreTrainedModel | nn.Module | None = None,
        sft_model: PreTrainedModel | nn.Module | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        callbacks: list[TrainerCallback] | None = None,
        **kwargs,
    ):
        self.data_collator = data_collator

        self.average_log_prob = args.average_log_prob
        self.sync_ref_settings = args.sync_ref_settings

        if hasattr(args, 'loss_settings'):
            self.loss_type = args.loss_settings['loss_type']  # type: ignore[index]

            if (
                self.loss_type in (DPOLossesType.SIMPO, DPOLossesType.ORPO, DPOLossesType.ASFT)
                and not args.average_log_prob
            ):
                raise ValueError(f'You should normalize logits by length when using {self.loss_type}')

            loss_args = args.loss_settings
            loss_args.pop('loss_type')  # type: ignore[union-attr]
            self.dpo_loss_registry = DPOLossRegistry.by_name(self.loss_type)(**loss_args)

        self._stored_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

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

        if hasattr(args, 'loss_settings') and self.loss_type in (
            DPOLossesType.SIMPO,
            DPOLossesType.ORPO,
            DPOLossesType.ASFT,
        ):
            logger.info(f'You can turn off ref_model when using {self.loss_type} for memory saving')

        self.ref_model = ref_model
        self.sft_model = sft_model

        if self.ref_model is not None:
            self.ref_model = prepare_model(self.ref_model, self.accelerator, self.is_deepspeed_enabled)

        if self.sft_model is not None:
            self.sft_model = prepare_model(self.sft_model, self.accelerator, self.is_deepspeed_enabled)

        default_callbacks = [DefaultFlowCallback] + get_reporting_integration_callbacks(self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = MetricsCallbackHandler(
            callbacks,
            model,
            tokenizer,
            None,
            None,
            ref_model=self.ref_model,
            sft_model=self.sft_model,
            accelerator=self.accelerator,
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else ProgressCallback)
        self.control: TrainerControl = self.callback_handler.on_init_end(self.args, self.state, self.control)

        if self.sync_ref_settings['sync_ref_model']:  # type: ignore[index]
            self.add_callback(SyncRefModelCallback(sync_ref_settings=self.sync_ref_settings))

    def dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        precomputed_margins: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.dpo_loss_registry.compute_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=reference_chosen_logps,
            reference_rejected_logps=reference_rejected_logps,
            precomputed_margins=precomputed_margins,
        )

    def _get_batch_logps(
        self,
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
        self, model: nn.Module, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        concatenated_batch = concatenated_inputs(batch, device=self.accelerator.device)

        precomputed_margins: torch.Tensor | None = concatenated_batch.pop('margin', None)

        all_logits = model(
            concatenated_batch['input_ids'],
            attention_mask=concatenated_batch['attention_mask'],
        ).logits.to(torch.float32)
        all_logps = self._get_batch_logps(
            all_logits,
            concatenated_batch['labels'],
            average_log_prob=self.average_log_prob,
        )
        chosen_idxs = batch['inputs_w']['input_ids'].shape[0]
        rejected_idx = batch['inputs_l']['input_ids'].shape[0]

        chosen_logps = all_logps[:chosen_idxs]
        rejected_logps = all_logps[chosen_idxs : chosen_idxs + rejected_idx]

        chosen_logits = all_logits[:chosen_idxs]
        rejected_logits = all_logits[chosen_idxs:]
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits, precomputed_margins

    def _get_logps(self, model: nn.Module | None, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if model is not None:
                (chosen_logps, rejected_logps, *_) = self.concatenated_forward(model, batch)
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    (
                        chosen_logps,
                        rejected_logps,
                        *_,
                    ) = self.concatenated_forward(self.model, batch)

        return chosen_logps, rejected_logps

    def get_batch_metrics(
        self,
        model: nn.Module,
        batch: dict[str, Any],
        train_eval: Literal['train', 'eval'] = 'train',
    ) -> tuple[torch.Tensor, dict[str, float]]:
        metrics: dict[str, float] = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            precomputed_margins,
        ) = self.concatenated_forward(model, batch)

        reference_chosen_logps, reference_rejected_logps = torch.Tensor([float('inf')]), torch.Tensor([float('inf')])

        if self.args.use_ref_model or self.loss_type not in (
            DPOLossesType.SIMPO,
            DPOLossesType.ORPO,
            DPOLossesType.ASFT,
        ):
            reference_chosen_logps, reference_rejected_logps = self._get_logps(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=reference_chosen_logps,
            reference_rejected_logps=reference_rejected_logps,
            precomputed_margins=precomputed_margins,
        )

        prefix = 'eval_' if train_eval == 'eval' else ''

        dpo_prefix_name = prefix + 'rewards/'

        metrics = self._compute_metrics(metrics, dpo_prefix_name, chosen_rewards, rejected_rewards)

        logp_accuracies = (policy_chosen_logps > policy_rejected_logps).float()
        metrics[f'{prefix}logps/accuracies'] = (logp_accuracies).detach().cpu().mean().item()
        metrics[f'{prefix}logps/rejected'] = (policy_rejected_logps).detach().cpu().mean().item()
        metrics[f'{prefix}logps/chosen'] = (policy_chosen_logps).detach().cpu().mean().item()

        metrics[f'{prefix}logits/rejected'] = (policy_rejected_logits).detach().cpu().mean().item()
        metrics[f'{prefix}logits/chosen'] = (policy_chosen_logits).detach().cpu().mean().item()

        if self.args.use_ref_model:
            ref_logp_accuracies = (reference_chosen_logps > reference_rejected_logps).float()
            metrics[f'{prefix}logps/ref_accuracies'] = (ref_logp_accuracies).detach().cpu().mean().item()
            metrics[f'{prefix}logps/ref_rejected'] = (reference_rejected_logps).detach().cpu().mean().item()
            metrics[f'{prefix}logps/ref_chosen'] = (reference_chosen_logps).detach().cpu().mean().item()

            metrics = self._compute_flips(
                metrics, prefix, logp_accuracies.detach().cpu(), ref_logp_accuracies.detach().cpu()
            )

        if self.loss_type == DPOLossesType.KTO:
            kto_chosen_KL = (
                (policy_chosen_logps.detach().cpu() - reference_chosen_logps.detach().cpu()).mean().clamp(min=0)
            )
            kto_rejected_KL = (
                (policy_rejected_logps.detach().cpu() - reference_rejected_logps.detach().cpu()).mean().clamp(min=0)
            )
            kto_z_chosen = self.dpo_loss_registry.beta * (chosen_rewards - kto_chosen_KL)
            kto_z_rejected = self.dpo_loss_registry.beta * (rejected_rewards - kto_rejected_KL)
            kto_grad_term_chosen = (-1 * F.sigmoid(kto_z_chosen) * F.sigmoid(-kto_z_chosen)).mean()
            kto_grad_term_rejected = (1 * F.sigmoid(kto_z_rejected) * F.sigmoid(-kto_z_rejected)).mean()
            kto_grad_term = kto_grad_term_chosen + kto_grad_term_rejected

            metrics[f'{prefix}rewards/kto_grad_term'] = kto_grad_term.item()
            metrics[f'{prefix}rewards/kto_grad_term_chosen'] = kto_grad_term_chosen.item()
            metrics[f'{prefix}rewards/kto_grad_term_rejected'] = kto_grad_term_rejected.item()

        elif self.loss_type == DPOLossesType.ORPO:
            log_odds = (policy_chosen_logps - policy_rejected_logps) - (
                torch.log1p(-torch.clamp(torch.exp(policy_chosen_logps), max=1 - 1e-7))
                - torch.log1p(-torch.clamp(torch.exp(policy_rejected_logps), max=1 - 1e-7))
            )
            ratio = -F.logsigmoid(log_odds)
            or_loss = self.dpo_loss_registry.beta * ratio
            nll_loss = -policy_chosen_logps

            metrics[f'{prefix}orpo/nll_loss'] = nll_loss.detach().cpu().mean().item()
            metrics[f'{prefix}orpo/or_loss'] = or_loss.detach().cpu().mean().item()
            metrics[f'{prefix}orpo/ratio'] = (ratio).detach().cpu().mean().item()
            metrics[f'{prefix}orpo/log_odds'] = (log_odds).detach().cpu().mean().item()

        elif self.loss_type == DPOLossesType.ASFT:
            chosen_ratio = policy_chosen_logps - (
                torch.log1p(-torch.clamp(torch.exp(policy_chosen_logps), max=1 - 1e-7))
            )
            rejected_ratio = policy_rejected_logps - (
                torch.log1p(-torch.clamp(torch.exp(policy_rejected_logps), max=1 - 1e-7))
            )
            chosen_logsig = -F.logsigmoid(chosen_ratio)
            rejected_logsig = -F.logsigmoid(-rejected_ratio)

            asft_loss = self.dpo_loss_registry.beta * (chosen_logsig + rejected_logsig)
            nll_loss = -policy_chosen_logps

            metrics[f'{prefix}asft/nll_loss'] = nll_loss.detach().cpu().mean().item()
            metrics[f'{prefix}asft/asft_loss'] = asft_loss.detach().cpu().mean().item()
            metrics[f'{prefix}asft/chosen_logsig'] = chosen_logsig.detach().cpu().mean().item()
            metrics[f'{prefix}asft/rejected_logsig'] = rejected_logsig.detach().cpu().mean().item()

        if self.sft_model is not None:
            sft_chosen_logps, sft_rejected_logps = self._get_logps(self.sft_model, batch)

            with torch.no_grad():
                _, sft_chosen_rewards, sft_rejected_rewards = self.dpo_loss(
                    policy_chosen_logps=policy_chosen_logps,
                    policy_rejected_logps=policy_rejected_logps,
                    reference_chosen_logps=sft_chosen_logps,
                    reference_rejected_logps=sft_rejected_logps,
                    precomputed_margins=precomputed_margins,
                )

            sft_prefix_name = prefix + 'rewards/sft_'
            metrics = self._compute_metrics(metrics, sft_prefix_name, sft_chosen_rewards, sft_rejected_rewards)

        return losses.mean(), metrics

    def _compute_metrics(
        self, metrics: dict[str, float], prefix_name: str, chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor
    ) -> dict[str, float]:
        accuracies = (chosen_rewards > rejected_rewards).float()
        metrics[f'{prefix_name}chosen'] = (chosen_rewards).detach().cpu().mean().item()
        metrics[f'{prefix_name}rejected'] = (rejected_rewards).detach().cpu().mean().item()
        metrics[f'{prefix_name}margins'] = (chosen_rewards - rejected_rewards).detach().cpu().mean().item()
        metrics[f'{prefix_name}accuracies'] = accuracies.detach().cpu().mean().item()

        metrics[f'{prefix_name}grad_term'] = (
            (self.dpo_loss_registry.beta * F.sigmoid(rejected_rewards - chosen_rewards)).detach().cpu().mean().item()
        )
        metrics[f'{prefix_name}grad_term_std'] = (
            (self.dpo_loss_registry.beta * F.sigmoid(rejected_rewards - chosen_rewards)).detach().cpu().std().item()
        )

        return metrics

    def _compute_flips(
        self,
        metrics: dict[str, Any],
        prefix_name: str,
        logp_accuracies: torch.Tensor,
        ref_logp_accuracies: torch.Tensor,
    ):
        correct_correct = (ref_logp_accuracies == 1) & (logp_accuracies == 1)
        correct_incorrect = (ref_logp_accuracies == 1) & (logp_accuracies == 0)
        incorrect_correct = (ref_logp_accuracies == 0) & (logp_accuracies == 1)
        incorrect_incorrect = (ref_logp_accuracies == 0) & (logp_accuracies == 0)

        correct_correct_count = correct_correct.sum().item()
        correct_incorrect_count = correct_incorrect.sum().item()
        incorrect_correct_count = incorrect_correct.sum().item()
        incorrect_incorrect_count = incorrect_incorrect.sum().item()

        total_count = len(logp_accuracies)

        correct_correct_ratio = correct_correct_count / total_count
        correct_incorrect_ratio = correct_incorrect_count / total_count
        incorrect_correct_ratio = incorrect_correct_count / total_count
        incorrect_incorrect_ratio = incorrect_incorrect_count / total_count

        metrics[f'{prefix_name}flips/correct->correct'] = correct_correct_ratio
        metrics[f'{prefix_name}flips/correct->incorrect'] = correct_incorrect_ratio
        metrics[f'{prefix_name}flips/incorrect->correct'] = incorrect_correct_ratio
        metrics[f'{prefix_name}flips/incorrect->incorrect'] = incorrect_incorrect_ratio

        return metrics

    def compute_loss(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs=False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        loss, metrics = self.get_batch_metrics(model, inputs, train_eval='train')

        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval='train')
        if return_outputs:
            return loss, metrics
        return loss

    def prediction_step(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if ignore_keys is None:
            if hasattr(model, 'config'):
                ignore_keys = getattr(model.config, 'keys_to_ignore_at_inference', [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, metrics = self.get_batch_metrics(model, inputs, train_eval='eval')

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

    def store_metrics(self, metrics: dict[str, float], train_eval: Literal['train', 'eval'] = 'train') -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: dict[str, float]) -> None:
        train_eval = 'train' if 'loss' in logs else 'eval'
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).cpu().mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)  # pylint: disable=no-member
