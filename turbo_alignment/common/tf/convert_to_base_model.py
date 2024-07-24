import torch
from peft import PeftModel

from turbo_alignment.common.tf.loaders import load_model, load_tokenizer
from turbo_alignment.settings.pipelines.common.merge_adapters_to_base import (
    MergeAdaptersToBaseModelSettings,
)


def peft_to_base_model(settings: MergeAdaptersToBaseModelSettings) -> None:
    tokenizer = load_tokenizer(settings.tokenizer_settings, settings.model_settings)
    base_model = load_model(settings.model_settings, tokenizer)

    first_weight = base_model.model.layers[0].self_attn.q_proj.weight
    first_weight_old = first_weight.clone()

    lora_model = PeftModel.from_pretrained(
        base_model,
        settings.adapter_path,
        device_map='auto',
        torch_dtype=base_model.dtype,
    )

    assert torch.allclose(first_weight_old, first_weight)

    lora_model = lora_model.merge_and_unload()
    lora_model.train(False)

    assert not torch.allclose(first_weight_old, first_weight)

    lora_model.save_pretrained(
        settings.save_path,
        max_shard_size=settings.max_shard_size,
    )
    tokenizer.save_pretrained(settings.save_path)
