from typing import Any

import torch

from turbo_alignment.dataset.chat.models import ChatDatasetRecord
from turbo_alignment.generators.base import ChatGeneratorBase
from turbo_alignment.modeling import parallel_states
from turbo_alignment.sequence_parallel.collator import pad_for_sequence_parallel, pad_and_slice
from turbo_alignment.settings.generators.outputs.chat import (
    AnswerMessage,
    ChatInferenceOutput,
)

import torch.distributed as dist
from turbo_alignment.dist_utils.order import run_in_order


def slice_tensor(tensor: torch.Tensor, world_size: int, rank: int, dim: int = -1) -> torch.Tensor:
    dim_size = tensor.size(dim)
    chunk_size = (dim_size + world_size - 1) // world_size  # round up
    actual_size = min(chunk_size, dim_size - chunk_size * rank)
    return tensor.narrow(dim, chunk_size * rank, actual_size)


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

        if parallel_states.sequence_parallel_is_initialized():
            batched_input_ids = pad_for_sequence_parallel(
                batched_input_ids,
                parallel_states.get_sequence_parallel_world_size(),
                padding_side=self._tokenizer.padding_side,
                padding_value=0,
            )

            batched_attention_mask = pad_for_sequence_parallel(
                batched_attention_mask,
                parallel_states.get_sequence_parallel_world_size(),
                padding_side=self._tokenizer.padding_side,
                padding_value=0,
            )

            seq_len = batched_input_ids.size(1)
            chunk_size = seq_len / parallel_states.get_sequence_parallel_world_size()
            rank = parallel_states.get_sequence_parallel_rank()
            batched_input_ids = batched_input_ids[:, rank * chunk_size : (rank + 1) * chunk_size]

            run_in_order()(print)(f'{dist.get_rank()=} {batched_input_ids.tolist()=}')

        else:
            print('WHAT')

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

        actual_input_ids = input_ids

        if parallel_states.sequence_parallel_is_initialized():
            run_in_order()(print)(f'Before {dist.get_rank()=} {input_ids.tolist()=}')
            actual_input_ids = slice_tensor(
                input_ids,
                parallel_states.get_sequence_parallel_world_size(),
                parallel_states.get_sequence_parallel_rank(),
                dim=-1,
            )

            # attention_mask = pad_for_sequence_parallel(
            #     attention_mask,
            #     parallel_states.get_sequence_parallel_world_size(),
            #     padding_side='left',
            #     padding_value=0,
            # )

            run_in_order()(print)(f'After {dist.get_rank()=} {input_ids.tolist()=}')

        else:
            print('WHAT')

        output_indices = self._model.generate(
            inputs=actual_input_ids,
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
                actual_output_indices = output_indices
                attention_mask = None
                if parallel_states.sequence_parallel_is_enabled():
                    attention_mask = torch.full_like(output_indices, fill_value=1)
                    attention_mask = pad_for_sequence_parallel(
                        attention_mask,
                        parallel_states.get_sequence_parallel_world_size(),
                        0,
                        padding_side='left',
                    )

                    actual_output_indices = pad_and_slice(
                        output_indices,
                        parallel_states.get_sequence_parallel_world_size(),
                        parallel_states.get_sequence_parallel_rank(),
                        self._tokenizer.pad_token_id,
                        padding_side='left',
                    )

                logits = self._model(actual_output_indices, attention_mask=attention_mask).logits.cpu()
                ws = parallel_states.get_sequence_parallel_world_size_or_one()
                assert logits.size(-2) == actual_output_indices.size(-1), (logits.size(), actual_output_indices.size())
                if ws != 1:
                    remainder = output_indices.size(1) % ws
                    padding = 0 if remainder == 0 else (ws - remainder)
                    if padding != 0:
                        logits = logits[:, padding:]

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
