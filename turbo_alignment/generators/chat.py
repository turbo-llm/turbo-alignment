from typing import Any

import torch
from transformers import BatchEncoding

from turbo_alignment.dataset.chat.models import ChatDatasetRecord
from turbo_alignment.generators.base import ChatGeneratorBase
from turbo_alignment.settings.generators.outputs.chat import (
    AnswerMessage,
    ChatInferenceOutput,
)


class ChatGenerator(ChatGeneratorBase[ChatDatasetRecord, ChatInferenceOutput]):
    def generate_from_batch_records(
        self,
        dataset_name: str,
        records_batch: dict[str, torch.Tensor] | BatchEncoding,
        original_records: list[ChatDatasetRecord] | None = None,
    ) -> list[ChatInferenceOutput]:
        batched_input_ids = records_batch['input_ids'].to(self.device)
        batched_attention_mask = records_batch['attention_mask'].to(self.device)

        assert self._tokenizer.padding_side == 'left'
        output_indices = self._model.generate(
            inputs=batched_input_ids,
            attention_mask=batched_attention_mask,
            generation_config=self._transformers_generator_parameters,
            tokenizer=self._tokenizer,
            pad_token_id=self._tokenizer.pad_token_id,
        )

        logits: torch.Tensor | None = None

        if self._custom_generation_settings.return_logits:
            with torch.no_grad():
                logits = self._model(output_indices).logits

        postprocessed_output_indices, answers_attention_mask, postprocessed_logits, answers = self._postprocess(
            input_indices=batched_input_ids,
            output_indices=output_indices,
            logits=logits,
            remove_prompt=self._custom_generation_settings.remove_prompt,
            only_answer_logits=self._custom_generation_settings.only_answer_logits,
        )

        outputs = []
        for i, (input_ids, attention_mask) in enumerate(zip(batched_input_ids, batched_attention_mask)):
            ans_batch_start = i * self._transformers_generator_parameters.num_return_sequences
            ans_batch_end = (i + 1) * self._transformers_generator_parameters.num_return_sequences
            batch_answer_tokens = postprocessed_output_indices[ans_batch_start:ans_batch_end, :]
            batch_answer_attention_masks = answers_attention_mask[ans_batch_start:ans_batch_end, :]
            batch_answers = answers[ans_batch_start:ans_batch_end]

            for j, (answer, answer_tokens, answer_attention_mask) in enumerate(
                zip(batch_answers, batch_answer_tokens, batch_answer_attention_masks)
            ):
                answer_logits = None if postprocessed_logits is None else postprocessed_logits[i + j, :, :]

                outputs.append(
                    ChatInferenceOutput(
                        dataset_name=dataset_name,
                        messages=original_records[i].messages if original_records else None,
                        label=original_records[i].label if original_records else None,
                        meta=original_records[i].meta if original_records else None,
                        input_token_ids=input_ids,
                        input_attention_mask=attention_mask,
                        answers=[
                            AnswerMessage(
                                id=str(i),
                                content=answer,
                                answer_token_ids=answer_tokens,
                                logits=answer_logits,
                                answer_attention_mask=answer_attention_mask,
                            )
                        ],
                    )
                )

        return outputs

    def generate_from_single_record(
        self,
        dataset_name: str,
        record: dict[str, Any],
        original_record: ChatDatasetRecord | None = None,
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

        logits: torch.Tensor | None = None

        if self._custom_generation_settings.return_logits:
            with torch.no_grad():
                logits = self._model(output_indices).logits.cpu()

        postprocessed_output_indices, answers_attention_mask, postprocessed_logits, answers = self._postprocess(
            input_indices=input_ids,
            output_indices=output_indices,
            logits=logits,
            remove_prompt=self._custom_generation_settings.remove_prompt,
            only_answer_logits=self._custom_generation_settings.only_answer_logits,
        )

        answer_messages = []
        for i, (answer, answer_tokens, answer_attention_mask) in enumerate(
            zip(answers, postprocessed_output_indices, answers_attention_mask)
        ):
            answer_logits = None if postprocessed_logits is None else postprocessed_logits[i, :, :].unsqueeze(0)
            answer_messages.append(
                AnswerMessage(
                    id=str(i),
                    content=answer,
                    answer_token_ids=answer_tokens.unsqueeze(0),
                    answer_attention_mask=answer_attention_mask,
                    logits=answer_logits,
                )
            )

        return ChatInferenceOutput(
            dataset_name=dataset_name,
            messages=original_record.messages,
            label=original_record.label,
            meta=original_record.meta,
            answers=answer_messages,
            input_token_ids=input_ids,
            input_attention_mask=attention_mask,
        )
