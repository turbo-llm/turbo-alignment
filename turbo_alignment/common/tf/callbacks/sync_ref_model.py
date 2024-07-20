from accelerate.utils import is_deepspeed_available
from torch import nn
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from turbo_alignment.settings.pipelines.train.dpo import SyncRefModelSettings


class SyncRefModelCallback(TrainerCallback):
    def __init__(self, *args, sync_ref_settings: SyncRefModelSettings, **kwargs) -> None:
        self._sync_ref_settings = sync_ref_settings

        super().__init__(*args, **kwargs)

    @staticmethod
    def _sync_target_model(model: nn.Module, target_model: nn.Module, alpha: float) -> None:
        for target_param, copy_param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_((alpha * copy_param.data) + (1.0 - alpha) * target_param.data)

    @classmethod
    def sync_target_model(
        cls, model: nn.Module, target_model: nn.Module, alpha: float, is_zero3_enabled: bool = False
    ) -> None:
        if is_deepspeed_available() and is_zero3_enabled:
            import deepspeed

            with deepspeed.zero.GatheredParameters(list(model.parameters()), modifier_rank=0):
                if deepspeed.comm.get_rank() == 0:
                    cls._sync_target_model(model, target_model, alpha)

        cls._sync_target_model(model, target_model, alpha)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        accelerator = kwargs.get('accelerator', None)
        ref_model = kwargs.get('ref_model', None)
        model = kwargs.get('model', None)

        is_zero3_enabled = args.deepspeed_plugin.zero_stage == 3 if args.deepspeed_plugin is not None else False

        if ref_model is not None and state.global_step % self._sync_ref_settings.sync_steps == 0:
            if accelerator is not None:
                unwrapped_model = accelerator.unwrap_model(model)
                self.sync_target_model(
                    model=unwrapped_model,
                    target_model=ref_model,
                    alpha=self._sync_ref_settings.alpha,
                    is_zero3_enabled=is_zero3_enabled,
                )
            else:
                self.sync_target_model(model=model, target_model=ref_model, alpha=self._sync_ref_settings.alpha)
