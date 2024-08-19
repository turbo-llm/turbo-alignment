from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import is_deepspeed_available
from allenai_common import Registrable
from torch import nn
from transformers import PreTrainedModel


def concatenated_inputs(
    batch: dict[str, Any], prefix: str = '', device: torch.device = torch.device('cpu')
) -> dict[str, torch.Tensor]:
    """
    Объединяем chosen и rejected инпуты в один батч, чтобы прогнать их за один forward
    """

    grouped_batch: dict[str, list[torch.Tensor]] = defaultdict(list)
    for outcome_key, outcome_inputs in batch.items():
        if outcome_key.startswith(prefix):
            for k, v in outcome_inputs.items():
                grouped_batch[k].append(v)

    concatenated_batch: dict[str, torch.Tensor] = {}
    for k, v in grouped_batch.items():
        concatenated_batch[k] = torch.cat(v, dim=0).to(device)

    return concatenated_batch


def prepare_model(
    model: PreTrainedModel | nn.Module, accelerator: Accelerator, is_deepspeed_enabled: bool = False
) -> PreTrainedModel | nn.Module:
    if is_deepspeed_enabled:
        model = prepare_model_for_deepspeed(model, accelerator)
    else:
        model = accelerator.prepare_model(model, evaluation_mode=True)

    return model


def prepare_model_for_deepspeed(
    model: PreTrainedModel | nn.Module, accelerator: Accelerator
) -> PreTrainedModel | nn.Module:
    if not is_deepspeed_available():
        raise ValueError('Deepspeed is not installed')

    import deepspeed

    deepspeed_plugin = accelerator.state.deepspeed_plugin
    config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
    if model is not None:
        if hasattr(model, 'config'):
            hidden_size: int | None = (
                max(model.config.hidden_sizes)
                if getattr(model.config, 'hidden_sizes', None)
                else getattr(model.config, 'hidden_size', None)
            )

            if hidden_size is not None and config_kwargs['zero_optimization']['stage'] == 3:
                config_kwargs.update(
                    {
                        'zero_optimization.reduce_bucket_size': hidden_size * hidden_size,
                        'zero_optimization.stage3_param_persistence_threshold': 10 * hidden_size,
                        'zero_optimization.stage3_prefetch_bucket_size': 0.9 * hidden_size * hidden_size,
                    }
                )

    if config_kwargs['zero_optimization']['stage'] != 3:
        config_kwargs['zero_optimization']['stage'] = 0
    config_kwargs['optimizer'] = {'type': None}

    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    return model


class DPOLossRegistry(Registrable):
    @abstractmethod
    def compute_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        policy_best_decode_logps: torch.FloatTensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...
