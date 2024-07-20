from typing import Any

from turbo_alignment.dataset.pair_preferences.collators import (
    PairPreferenceDataCollator,
)


class KTODataCollator(PairPreferenceDataCollator):
    def __call__(self, examples: list[dict[str, dict[str, Any]]]) -> dict[str, Any]:
        max_length = 0
        for ex in examples:
            max_length = max(max_length, len(ex['chat']['input_ids']))
            max_length = max(max_length, len(ex['KL']['input_ids']))

        chat_batch = dict(self._get_batch(examples, self.tokenizer, 'chat', max_length))
        kl_batch = dict(self._get_batch(examples, self.tokenizer, 'KL', max_length))
        desirable_batch = [ex['is_desirable'] for ex in examples]

        return {
            'is_desirable': desirable_batch,
            'input_ids': chat_batch['input_ids'],
            'attention_mask': chat_batch['attention_mask'],
            'labels': chat_batch['labels'],
            'KL_input_ids': kl_batch['input_ids'],
            'KL_attention_mask': kl_batch['attention_mask'],
            'KL_labels': kl_batch['labels'],
        }
