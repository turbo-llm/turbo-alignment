from dataclasses import dataclass
from typing import Any

from transformers import PreTrainedTokenizerBase

from turbo_alignment.dataset.pair_preferences import PairPreferenceDataCollator


@dataclass
class DDPODataCollator(PairPreferenceDataCollator):
    rm_tokenizer: PreTrainedTokenizerBase = None

    def __call__(self, examples: list[dict[str, dict[str, Any]]]) -> dict[str, Any]:
        sft_max_length, rm_max_length = 0, 0
        for ex in examples:
            sft_max_length = max(sft_max_length, len(ex['sft_inputs_w']['input_ids']))
            sft_max_length = max(sft_max_length, len(ex['sft_inputs_l']['input_ids']))
            rm_max_length = max(rm_max_length, len(ex['rm_inputs_w']['input_ids']))
            rm_max_length = max(rm_max_length, len(ex['rm_inputs_l']['input_ids']))

        batch = {
            'sft_inputs_w': dict(self._get_batch(examples, self.tokenizer, 'sft_inputs_w', sft_max_length)),
            'sft_inputs_l': dict(self._get_batch(examples, self.tokenizer, 'sft_inputs_l', sft_max_length)),
            'rm_inputs_w': dict(self._get_batch(examples, self.rm_tokenizer, 'rm_inputs_w', rm_max_length)),
            'rm_inputs_l': dict(self._get_batch(examples, self.rm_tokenizer, 'rm_inputs_l', rm_max_length)),
        }

        return batch
