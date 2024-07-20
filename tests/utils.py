import torch
from transformers import PreTrainedTokenizerBase


def is_sample_build_from_content(
    input_ids: torch.Tensor, contents: list[str], tokenizer: PreTrainedTokenizerBase
) -> bool:
    decoded_sample = tokenizer.decode(input_ids, skip_special_tokens=True)
    for c in contents:
        if c not in decoded_sample:
            return False
        decoded_sample = decoded_sample.replace(c, '', 1)

    return True
