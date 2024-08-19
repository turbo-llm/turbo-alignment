from typing import Any

import torch

from turbo_alignment.dataset.chat.models import ChatDatasetRecord
from turbo_alignment.generators.base import ChatGeneratorBase
from turbo_alignment.settings.generators.outputs.chat import (
    AnswerMessage,
    ChatInferenceOutput,
)


class ChatGenerator(ChatGeneratorBase[ChatDatasetRecord, ChatInferenceOutput]):
    def _generate_from_batch_records(
        self,
        records: list[dict[str, torch.Tensor]],
        original_records: list[ChatDatasetRecord],
        dataset_name: str,
    ) -> list[ChatInferenceOutput]:
        input_ids = [record['input_ids'].tolist() for record in records]
        attention_mask = [record['attention_mask'].tolist() for record in records]

        max_input_length = max(len(sample) for sample in input_ids)

        batch = self._tokenizer.pad(
            {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            },
            padding='max_length',
            max_length=max_input_length,
            return_tensors='pt',
        )

        batched_input_ids = batch['input_ids'].to(self.device)
        batched_attention_mask = batch['attention_mask'].to(self.device)

        output_indices = self._model.generate(
            inputs=batched_input_ids,
            attention_mask=batched_attention_mask,
            generation_config=self._transformers_generator_parameters,
            tokenizer=self._tokenizer,
            pad_token_id=self._tokenizer.pad_token_id,
        )

        postprocessed_output_indices = self._postprocess(
            input_indices=batched_input_ids,
            output_indices=output_indices,
            remove_prompt=self._custom_generation_settings.remove_prompt,
        )

        answers = self._decode(token_indices=postprocessed_output_indices)

        return [
            ChatInferenceOutput(
                id=original_record.id,
                dataset_name=dataset_name,
                messages=original_record.messages,
                label=original_record.label,
                meta=original_record.meta,
                answers=[
                    AnswerMessage(
                        id='0',
                        content=answer,
                        input_token_ids=None,
                        answer_token_ids=None,
                        logits=None,
                    )
                ],
            )
            for original_record, answer in zip(original_records, answers)
        ]

    def _generate_from_single_record(
        self,
        record: dict[str, Any],
        original_record: ChatDatasetRecord,
        dataset_name: str,
    ) -> ChatInferenceOutput:
        input_ids = torch.unsqueeze(record['input_ids'], 0).to(self.device)
        attention_mask = torch.unsqueeze(record['attention_mask'], 0).to(self.device)

        output_indices = self._model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            generation_config=self._transformers_generator_parameters,
            tokenizer=self._tokenizer,
            pad_token_id=self._tokenizer.pad_token_id,
        )

        postprocessed_output_indices = self._postprocess(
            input_indices=input_ids,
            output_indices=output_indices,
            remove_prompt=self._custom_generation_settings.remove_prompt,
        )

        answers = self._decode(token_indices=postprocessed_output_indices)

        logits: torch.Tensor | None = None
        input_token_ids: torch.Tensor | None = None
        answer_tokens_ids: torch.Tensor | None = None

        if self._return_logits:
            with torch.no_grad():
                logits = self._model(output_indices).logits.cpu()

            answer_tokens_ids = postprocessed_output_indices
            input_token_ids = input_ids

            answer_messages = [
                AnswerMessage(
                    id=str(i),
                    content=a,
                    input_token_ids=input_token_ids,
                    answer_token_ids=a_t_ids.unsqueeze(0),
                    logits=l.unsqueeze(0),
                )
                for i, (a, a_t_ids, l) in enumerate(zip(answers, answer_tokens_ids, logits))  # type: ignore[arg-type]
            ]
        else:
            answer_messages = [
                AnswerMessage(
                    id=str(i),
                    content=a,
                    input_token_ids=input_token_ids,
                    answer_token_ids=answer_tokens_ids,
                    logits=logits,
                )
                for i, a in enumerate(answers)
            ]

        return ChatInferenceOutput(
            id=original_record.id,
            dataset_name=dataset_name,
            messages=original_record.messages,
            label=original_record.label,
            meta=original_record.meta,
            answers=answer_messages,
        )
