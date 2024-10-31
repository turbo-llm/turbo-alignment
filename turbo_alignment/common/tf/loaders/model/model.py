import torch
from peft import PeftModel, get_peft_model, prepare_model_for_int8_training
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from turbo_alignment.common.tf.loaders.model.registry import (
    PeftConfigRegistry,
    TransformersAutoModelRegistry,
)
from turbo_alignment.modeling.liger_kernels import apply_liger_kernel_to_gemma2
from turbo_alignment.settings.model import (
    ModelForPeftSettings,
    ModelType,
    PreTrainedAdaptersModelSettings,
    PreTrainedModelSettings,
)
from turbo_alignment.settings.tf.peft import PEFT_TYPE


def _prepare_model_for_peft(model: PreTrainedModel, peft_settings: PEFT_TYPE) -> PeftModel:
    peft_params = peft_settings.dict()
    peft_params.pop('name')

    peft_config = PeftConfigRegistry.by_name(peft_settings.name)(**peft_params)

    return get_peft_model(model, peft_config)


def _load_pretrained_adapters(
    model: PreTrainedModel,
    model_settings: PreTrainedAdaptersModelSettings,
) -> PeftModel:
    return PeftModel.from_pretrained(
        model,
        model_settings.adapter_path,
        is_trainable=model_settings.is_trainable,
    )


def unfreeze_params(layer):
    for param in layer.parameters():
        param.requires_grad = True


def load_model(
    model_settings: PreTrainedModelSettings,
    tokenizer: PreTrainedTokenizerBase,
) -> PreTrainedModel:
    if model_settings.liger_kernels_settings is not None:
        apply_liger_kernel_to_gemma2(
            rope=model_settings.liger_kernels_settings.use_rope,
            cross_entropy=model_settings.liger_kernels_settings.use_cross_entropy,
            geglu=model_settings.liger_kernels_settings.use_geglu,
        )

    model = TransformersAutoModelRegistry.by_name(model_settings.model_type).from_pretrained(
        model_settings.model_path,
        **model_settings.transformers_settings.dict(exclude_none=True),
        **model_settings.model_kwargs,
        torch_dtype=torch.bfloat16,
    )

    if model_settings.transformers_settings.load_in_8bit:
        model = prepare_model_for_int8_training(model)

    model.resize_token_embeddings(len(tokenizer))

    if model_settings.embeddings_initialization_strategy is not None:
        with torch.no_grad():
            for new_token, old_token in model_settings.embeddings_initialization_strategy.items():
                new_token_id = tokenizer.get_added_vocab()[new_token]
                old_token_id = tokenizer.encode(old_token, add_special_tokens=False)[0]

                if model.config.model_type == 'gpt_neox':
                    model.gpt_neox.embed_in.weight[new_token_id, :] = torch.clone(
                        model.gpt_neox.embed_in.weight[old_token_id, :]
                    )
                    if model_settings.model_type == 'causal':
                        model.embed_out.weight[new_token_id, :] = torch.clone(model.embed_out.weight[old_token_id, :])

                elif model.config.model_type == 'llama':
                    model.model.embed_tokens.weight[new_token_id, :] = model.model.embed_tokens.weight[old_token_id, :]

    if isinstance(model_settings, PreTrainedAdaptersModelSettings):
        model = _load_pretrained_adapters(model, model_settings)
    elif isinstance(model_settings, ModelForPeftSettings):
        # creating learnable adapters and freezing non-training parameters
        model = _prepare_model_for_peft(model, model_settings.peft_settings)

        # deepspeed stage3 is currently doens't work with seq_cls head and peft
        if model_settings.model_type == ModelType.SEQ_CLS and is_deepspeed_zero3_enabled():
            model.base_model.model.score = torch.nn.Linear(
                in_features=model.base_model.model.score.original_module.in_features,
                out_features=model.base_model.model.score.original_module.out_features,
                bias=model.base_model.model.score.original_module.bias,
            )
            model.base_model.model.score.weight.requires_grad = True

    return model
