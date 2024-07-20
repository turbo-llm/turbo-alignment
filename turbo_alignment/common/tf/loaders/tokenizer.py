from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from turbo_alignment.settings.model import PreTrainedModelSettings
from turbo_alignment.settings.tf.tokenizer import TokenizerSettings


def load_tokenizer(
    tokenizer_settings: TokenizerSettings,
    model_settings: PreTrainedModelSettings,
) -> PreTrainedTokenizerBase:
    tokenizer_path = tokenizer_settings.tokenizer_path or model_settings.model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=tokenizer_settings.use_fast,
        trust_remote_code=tokenizer_settings.trust_remote_code,
        **tokenizer_settings.tokenizer_kwargs,
    )
    return tokenizer
