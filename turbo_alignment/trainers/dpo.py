from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    BaseImageProcessor,
    DefaultFlowCallback,
    FeatureExtractionMixin,
    PreTrainedTokenizerBase,
    PrinterCallback,
    ProcessorMixin,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.modeling_utils import PreTrainedModel

from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.common.tf.callbacks.common import MetricsCallbackHandler
from turbo_alignment.common.tf.callbacks.sync_ref_model import SyncRefModelCallback
from turbo_alignment.constants import DISABLE_LOSS_LABEL
from turbo_alignment.settings.pipelines.train.dpo import (
    APODownLossSettings,
    APOZeroLossSettings,
    ASFTLossSettings,
    CALDPOLossSettings,
    CPOLossSettings,
    DPOLossesType,
    DPOPLossSettings,
    HingeLossSettings,
    IPOLossSettings,
    KTOLossSettings,
    NCAPairLossSettings,
    ORPOLossSettings,
    SigmoidLossSettings,
    SigmoidLossWithMarginSettings,
    SimPOLossSettings,
    SlicHfLossSettings,
    SyncRefModelSettings,
)
from turbo_alignment.trainers.utils import (
    DPOLossRegistry,
    prepare_model,
)
from turbo_alignment.modeling import parallel_states
from turbo_alignment.sequence_parallel.trainer import TrainerWithSeqP
from .base_args import TrainingArgumentsWithSeqP

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
    def __init__(self, *args, beta: float = 1.0, lambda_: float = 1.0, ce_coef: float = 1.0, **kwargs) -> None:
        self.beta = beta
        self.ce_coef = ce_coef
        self.lambda_ = lambda_
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

        ratio = -F.logsigmoid(self.beta * log_odds)
        losses = -(self.ce_coef * policy_chosen_logps) + self.lambda_ * ratio

        chosen_rewards = self.beta * policy_chosen_logps.detach()
        rejected_rewards = self.beta * policy_rejected_logps.detach()

        return losses, chosen_rewards, rejected_rewards


@DPOLossRegistry.register(DPOLossesType.ASFT)
class ASFTLoss(DPOLossRegistry):
    def __init__(self, *args, beta: float = 1.0, lambda_: float = 1.0, ce_coef: float = 1.0, **kwargs) -> None:
        self.beta = beta
        self.lambda_ = lambda_
        self.ce_coef = ce_coef
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
        sigm_diff = -F.logsigmoid(self.beta * chosen_ratio) - F.logsigmoid(-self.beta * rejected_ratio)

        losses = -(self.ce_coef * policy_chosen_logps) + self.lambda_ * sigm_diff

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


@DPOLossRegistry.register(DPOLossesType.DPOP)
class DPOPLoss(DPOLossRegistry):
    def __init__(self, *args, beta: float = 0.1, lam: float = 0.1, **kwargs) -> None:
        self.beta = beta
        self.lam = lam
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
        penalty_term = self.lam * torch.relu(reference_chosen_logps - policy_chosen_logps)

        logits = pi_logratios - ref_logratios

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps - penalty_term).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        loss = -F.logsigmoid(self.beta * (logits - penalty_term))

        return loss, chosen_rewards, rejected_rewards


@DPOLossRegistry.register(DPOLossesType.NCA_PAIR)
class NCAPairLoss(DPOLossRegistry):
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

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        loss = (
            -F.logsigmoid(chosen_logratios * self.beta)
            - 0.5 * F.logsigmoid(-chosen_logratios * self.beta)
            - 0.5 * F.logsigmoid(-rejected_logratios * self.beta)
        )

        return loss, chosen_rewards, rejected_rewards


@DPOLossRegistry.register(DPOLossesType.CAL_DPO)
class CALDPOLoss(DPOLossRegistry):
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
        chosen_reward = policy_chosen_logps - reference_chosen_logps
        reject_reward = policy_rejected_logps - reference_rejected_logps

        dpo_losses = -F.logsigmoid(chosen_reward - reject_reward)

        cal_losses = F.mse_loss(chosen_reward, torch.full_like(chosen_reward, 0.5 * (1 / self.beta))) + F.mse_loss(
            reject_reward, torch.full_like(reject_reward, -0.5 * (1 / self.beta))
        )
        loss = dpo_losses + cal_losses

        chosen_rewards = (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards


@dataclass
class DPOTrainingArguments(TrainingArgumentsWithSeqP):
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
        | DPOPLossSettings
        | NCAPairLossSettings
        | CALDPOLossSettings
    ) = field(
        default_factory=SigmoidLossSettings(loss_type=DPOLossesType.SIGMOID)
    )  # type: ignore[call-overload]
    sync_ref_settings: SyncRefModelSettings = field(  # type: ignore[call-overload]
        default_factory=SyncRefModelSettings()
    )
    use_ref_model: bool = True
    use_sft_model: bool = False
    average_log_prob: bool = False


class DPOTrainer(TrainerWithSeqP):
    """
    DPO trainer using sequential packing: [context | chosen | rejected].

    Mirrors `RMTrainer` style: a single forward pass over packed inputs, exact
    segment log-prob extraction at boundaries, and a configurable loss via
    DPOLossRegistry. Supports ref_model, sft_model, sequence parallelism,
    average_log_prob, precomputed_margins, and KTO/ORPO/ASFT specific metrics.
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
        processing_class: (
            PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin | None
        ) = None,
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

            loss_args = (
                args.loss_settings.model_dump()
                if hasattr(args.loss_settings, 'model_dump')
                else dict(args.loss_settings)
            )
            loss_args.pop('loss_type', None)
            self.dpo_loss_registry = DPOLossRegistry.by_name(self.loss_type)(**loss_args)

        self._stored_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
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
            self.ref_model = prepare_model(  # type: ignore[arg-type,assignment]
                self.ref_model, self.accelerator, self.is_deepspeed_enabled
            )

        if self.sft_model is not None:
            self.sft_model = prepare_model(  # type: ignore[arg-type,assignment]
                self.sft_model, self.accelerator, self.is_deepspeed_enabled
            )

        default_callbacks = [DefaultFlowCallback] + get_reporting_integration_callbacks(self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = MetricsCallbackHandler(
            callbacks,
            model,
            processing_class,
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

    # ------------------------------------------------------------------
    # Loss dispatch (configurable via DPOLossRegistry)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Core helpers (RM-style, sequential packing)
    # ------------------------------------------------------------------
    def _forward_logits(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask_4d: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run model forward over packed [context | chosen | rejected].

        Follows RM-style behavior: sequence-parallel padding/splitting is handled
        upstream by DataCollatorForSequenceParallism, so trainer does not pad/slice.
        """
        model_dtype = next(model.parameters()).dtype
        attn_additive = torch.finfo(model_dtype).min * (attention_mask_4d == 0).to(model_dtype)
        return model(
            input_ids=input_ids,
            attention_mask=attn_additive,
            position_ids=position_ids,
            use_cache=False,
        ).logits.to(torch.float32)

    def _extract_segment_logps(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        segment_start_indices: torch.Tensor,
        segment_end_indices: torch.Tensor,
        average_log_prob: bool = False,
    ) -> torch.Tensor:
        """
        Sum (or average) log-probabilities for tokens in
        (segment_start_indices, segment_end_indices].

        SP behavior mirrors RM: inputs are already rank-local (split in collator),
        then per-rank segment sums are all-reduced across sequence-parallel group.
        """
        # Local next-token shift
        local_logits = logits[:, :-1, :]
        local_labels = labels[:, 1:].clone()

        loss_mask = local_labels != DISABLE_LOSS_LABEL
        labels_masked = local_labels.clone()
        labels_masked[labels_masked == DISABLE_LOSS_LABEL] = 0

        per_token_logps = (
            torch.gather(
                local_logits.log_softmax(-1),
                dim=2,
                index=labels_masked.unsqueeze(2),
            ).squeeze(2)
            * loss_mask.float()
        )

        bsz, local_len = per_token_logps.shape

        if parallel_states.sequence_parallel_is_initialized():
            rank = parallel_states.get_sequence_parallel_rank()
            # logits length before local shift corresponds to local chunk size
            seq_len_chunk = logits.size(1)
            offset = rank * seq_len_chunk
            positions = (
                torch.arange(
                    offset,
                    offset + local_len,
                    device=per_token_logps.device,
                )
                .unsqueeze(0)
                .expand(bsz, -1)
            )

            seg_mask = (positions > segment_start_indices.unsqueeze(1)) & (
                positions <= segment_end_indices.unsqueeze(1)
            )

            local_sum = (per_token_logps * seg_mask.float()).sum(dim=1)
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM, group=parallel_states.get_sequence_parallel_group())

            if average_log_prob:
                local_n = seg_mask.sum(dim=1).float()
                dist.all_reduce(local_n, op=dist.ReduceOp.SUM, group=parallel_states.get_sequence_parallel_group())
                return torch.where(local_n > 0, local_sum / local_n, local_sum)

            return local_sum

        positions = torch.arange(local_len, device=per_token_logps.device).unsqueeze(0).expand(bsz, -1)
        seg_mask = (positions > segment_start_indices.unsqueeze(1)) & (positions <= segment_end_indices.unsqueeze(1))

        seg_sum = (per_token_logps * seg_mask.float()).sum(dim=1)
        if average_log_prob:
            seg_n = seg_mask.sum(dim=1).float()
            return torch.where(seg_n > 0, seg_sum / seg_n, seg_sum)
        return seg_sum

    def _segment_logps_for_model(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask_4d: torch.Tensor,
        position_ids: torch.Tensor,
        labels: torch.Tensor,
        context_end_indices: torch.Tensor,
        chosen_indices: torch.Tensor,
        rejected_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self._forward_logits(model, input_ids, attention_mask_4d, position_ids)
        chosen_logps = self._extract_segment_logps(
            logits,
            labels,
            context_end_indices,
            chosen_indices,
            average_log_prob=self.average_log_prob,
        )
        rejected_logps = self._extract_segment_logps(
            logits,
            labels,
            chosen_indices,
            rejected_indices,
            average_log_prob=self.average_log_prob,
        )
        return chosen_logps, rejected_logps, logits

    # ------------------------------------------------------------------
    # Main loss / metrics
    # ------------------------------------------------------------------
    def get_batch_metrics(
        self,
        model: nn.Module,
        batch: dict[str, Any],
        train_eval: Literal['train', 'eval'] = 'train',
    ) -> tuple[torch.Tensor, dict[str, float]]:
        device = self.accelerator.device

        input_ids = batch['input_ids'].to(device)
        attention_mask_4d = batch['attention_mask'].to(device)
        position_ids = batch['position_ids'].to(device)
        context_end_indices = batch['context_end_indices'].to(device)
        chosen_indices = batch['chosen_indices'].to(device)
        rejected_indices = batch['rejected_indices'].to(device)
        labels = input_ids.clone()

        precomputed_margins = batch.get('precomputed_margin')
        if precomputed_margins is not None:
            precomputed_margins = precomputed_margins.to(device)

        # ---- Policy ----
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_logits,
        ) = self._segment_logps_for_model(
            model,
            input_ids,
            attention_mask_4d,
            position_ids,
            labels,
            context_end_indices,
            chosen_indices,
            rejected_indices,
        )

        policy_chosen_logits = policy_logits.mean()
        policy_rejected_logits = policy_logits.mean()

        # ---- Reference ----
        needs_reference = self.args.use_ref_model or self.loss_type not in (  # type: ignore[attr-defined]
            DPOLossesType.SIMPO,
            DPOLossesType.ORPO,
            DPOLossesType.ASFT,
        )

        if needs_reference:
            if self.ref_model is not None:
                with torch.no_grad():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                    ) = self._segment_logps_for_model(
                        self.ref_model,
                        input_ids,
                        attention_mask_4d,
                        position_ids,
                        labels,
                        context_end_indices,
                        chosen_indices,
                        rejected_indices,
                    )
            else:
                # LoRA-style ref via adapter disable on the policy model
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                if hasattr(unwrapped_model, 'disable_adapter'):
                    disable_ctx = unwrapped_model.disable_adapter()
                elif hasattr(unwrapped_model, 'disable_adapters'):
                    disable_ctx = unwrapped_model.disable_adapters()
                else:
                    raise ValueError(
                        f'Loss type {self.loss_type} requires a reference model, '
                        f'but ref_model is None and the policy has no adapter to disable.'
                    )
                with torch.no_grad():
                    with disable_ctx:
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                        ) = self._segment_logps_for_model(
                            self.model,
                            input_ids,
                            attention_mask_4d,
                            position_ids,
                            labels,
                            context_end_indices,
                            chosen_indices,
                            rejected_indices,
                        )
        else:
            reference_chosen_logps = torch.zeros_like(policy_chosen_logps)
            reference_rejected_logps = torch.zeros_like(policy_rejected_logps)

        # ---- Loss ----
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=reference_chosen_logps,
            reference_rejected_logps=reference_rejected_logps,
            precomputed_margins=precomputed_margins,
        )

        # ---- Metrics ----
        prefix = 'eval_' if train_eval == 'eval' else ''
        metrics: dict[str, float] = {}
        metrics = self._compute_metrics(metrics, prefix + 'rewards/', chosen_rewards, rejected_rewards)

        logp_accuracies = (policy_chosen_logps > policy_rejected_logps).float()
        metrics[f'{prefix}logps/accuracies'] = logp_accuracies.detach().cpu().mean().item()
        metrics[f'{prefix}logps/rejected'] = policy_rejected_logps.detach().cpu().mean().item()
        metrics[f'{prefix}logps/chosen'] = policy_chosen_logps.detach().cpu().mean().item()
        metrics[f'{prefix}logits/rejected'] = policy_rejected_logits.detach().cpu().mean().item()
        metrics[f'{prefix}logits/chosen'] = policy_chosen_logits.detach().cpu().mean().item()

        if self.args.use_ref_model:  # type: ignore[attr-defined]
            ref_logp_accuracies = (reference_chosen_logps > reference_rejected_logps).float()
            metrics[f'{prefix}logps/ref_accuracies'] = ref_logp_accuracies.detach().cpu().mean().item()
            metrics[f'{prefix}logps/ref_rejected'] = reference_rejected_logps.detach().cpu().mean().item()
            metrics[f'{prefix}logps/ref_chosen'] = reference_chosen_logps.detach().cpu().mean().item()
            metrics = self._compute_flips(
                metrics,
                prefix,
                logp_accuracies.detach().cpu(),
                ref_logp_accuracies.detach().cpu(),
            )

        # Loss-specific extra metrics
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
            ratio = -F.logsigmoid(self.dpo_loss_registry.beta * log_odds)
            or_loss = self.dpo_loss_registry.lambda_ * ratio
            nll_loss = -(self.dpo_loss_registry.ce_coef * policy_chosen_logps)

            metrics[f'{prefix}orpo/nll_loss'] = nll_loss.detach().cpu().mean().item()
            metrics[f'{prefix}orpo/or_loss'] = or_loss.detach().cpu().mean().item()
            metrics[f'{prefix}orpo/ratio'] = ratio.detach().cpu().mean().item()
            metrics[f'{prefix}orpo/log_odds'] = log_odds.detach().cpu().mean().item()

        elif self.loss_type == DPOLossesType.ASFT:
            chosen_ratio = policy_chosen_logps - (
                torch.log1p(-torch.clamp(torch.exp(policy_chosen_logps), max=1 - 1e-7))
            )
            rejected_ratio = policy_rejected_logps - (
                torch.log1p(-torch.clamp(torch.exp(policy_rejected_logps), max=1 - 1e-7))
            )
            chosen_logsig = -F.logsigmoid(self.dpo_loss_registry.beta * chosen_ratio)
            rejected_logsig = -F.logsigmoid(-self.dpo_loss_registry.beta * rejected_ratio)

            asft_loss = self.dpo_loss_registry.lambda_ * (chosen_logsig + rejected_logsig)
            nll_loss = -(self.dpo_loss_registry.ce_coef * policy_chosen_logps)

            metrics[f'{prefix}asft/nll_loss'] = nll_loss.detach().cpu().mean().item()
            metrics[f'{prefix}asft/asft_loss'] = asft_loss.detach().cpu().mean().item()
            metrics[f'{prefix}asft/chosen_logsig'] = chosen_logsig.detach().cpu().mean().item()
            metrics[f'{prefix}asft/rejected_logsig'] = rejected_logsig.detach().cpu().mean().item()

        # ---- SFT model branch ----
        if self.sft_model is not None:
            with torch.no_grad():
                (
                    sft_chosen_logps,
                    sft_rejected_logps,
                    _,
                ) = self._segment_logps_for_model(
                    self.sft_model,
                    input_ids,
                    attention_mask_4d,
                    position_ids,
                    labels,
                    context_end_indices,
                    chosen_indices,
                    rejected_indices,
                )

                _, sft_chosen_rewards, sft_rejected_rewards = self.dpo_loss(
                    policy_chosen_logps=policy_chosen_logps,
                    policy_rejected_logps=policy_rejected_logps,
                    reference_chosen_logps=sft_chosen_logps,
                    reference_rejected_logps=sft_rejected_logps,
                    precomputed_margins=precomputed_margins,
                )

            metrics = self._compute_metrics(metrics, prefix + 'rewards/sft_', sft_chosen_rewards, sft_rejected_rewards)

        return losses.mean() / parallel_states.get_sequence_parallel_world_size_or_one(), metrics

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------
    def _compute_metrics(
        self,
        metrics: dict[str, float],
        prefix_name: str,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
    ) -> dict[str, float]:
        accuracies = (chosen_rewards > rejected_rewards).float()
        metrics[f'{prefix_name}chosen'] = chosen_rewards.detach().cpu().mean().item()
        metrics[f'{prefix_name}rejected'] = rejected_rewards.detach().cpu().mean().item()
        metrics[f'{prefix_name}margins'] = (chosen_rewards - rejected_rewards).detach().cpu().mean().item()
        metrics[f'{prefix_name}accuracies'] = accuracies.detach().cpu().mean().item()
        metrics[f'{prefix_name}grad_term'] = (
            (self.dpo_loss_registry.beta * F.sigmoid(rejected_rewards - chosen_rewards)).detach().cpu().mean().item()
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

        total_count = max(len(logp_accuracies), 1)
        metrics[f'{prefix_name}flips/correct->correct'] = correct_correct.sum().item() / total_count
        metrics[f'{prefix_name}flips/correct->incorrect'] = correct_incorrect.sum().item() / total_count
        metrics[f'{prefix_name}flips/incorrect->correct'] = incorrect_correct.sum().item() / total_count
        metrics[f'{prefix_name}flips/incorrect->incorrect'] = incorrect_incorrect.sum().item() / total_count
        return metrics

    # ------------------------------------------------------------------
    # HF Trainer hooks
    # ------------------------------------------------------------------
    def compute_loss(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs=False,
        num_items_in_batch=None,  # pylint: disable=unused-argument
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
            'logits_test/chosen': metrics.get('logits_test/chosen', metrics.get('eval_logits/chosen', 0.0)),
            'logits_test/rejected': metrics.get('logits_test/rejected', metrics.get('eval_logits/rejected', 0.0)),
        }
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.tensor(logits)
        labels = torch.zeros(logits.shape[0])

        return loss.detach(), logits, labels

    def store_metrics(self, metrics: dict[str, float], train_eval: Literal['train', 'eval'] = 'train') -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: dict[str, float], _start_time: float | None = None) -> None:
        train_eval = 'train' if 'loss' in logs else 'eval'
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).cpu().mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)  # pylint: disable=no-member
