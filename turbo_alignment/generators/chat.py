from typing import Any
import torch
from transformers import BatchEncoding

from turbo_alignment.dataset.chat.models import ChatDatasetRecord
from turbo_alignment.generators.base import ChatGeneratorBase
from turbo_alignment.settings.generators.outputs.chat import (
    AnswerMessage,
    ChatInferenceOutput,
)
from turbo_alignment.generators.base import BaseGenerator
from turbo_alignment.settings.tf.generation import GeneratorTransformersSettings
from turbo_alignment.settings.generators.chat import CustomChatGenerationSettings
from transformers import PreTrainedTokenizerBase
from vllm.lora.request import LoRARequest
from vllm import SamplingParams
import ray

class vLLMChatGenerator(BaseGenerator[ChatDatasetRecord, ChatInferenceOutput]):
    def __init__(
        self,
        transformers_settings: GeneratorTransformersSettings,
        custom_generation_settings: CustomChatGenerationSettings,
        vllm_engines: list,
        tokenizer: PreTrainedTokenizerBase,
        return_logits: bool = False,
        lora_request: LoRARequest | None = None,
    ):
        super().__init__(vllm_engines, tokenizer)

        if isinstance(transformers_settings.stop_strings, list):
            raise ValueError('You should use only 1 eos token with VLLM')

        #TODO separate stop_strings and eos_token
        self.eos_token_id: int = self._tokenizer.encode(transformers_settings.stop_strings, add_special_tokens=False)
        assert len(self.eos_token_id) == 1, 'Currently stop_strings == stop_token'
        #TODO beam search was deprecated in vllm 0.6.2

        # beam_search_params: dict[str, Any] = {
        #     'best_of': transformers_settings.num_return_sequences,
        #     'use_beam_search': False,
        # }
        # if transformers_settings.num_beams > 1:
        #     beam_search_params['use_beam_search'] = True
        #     beam_search_params['best_of'] = transformers_settings.num_beams
        
        self._sampling_params = SamplingParams(
            n=transformers_settings.num_return_sequences,
            repetition_penalty=transformers_settings.repetition_penalty,
            temperature=transformers_settings.temperature,
            top_p=transformers_settings.top_p,
            top_k=transformers_settings.top_k,
            skip_special_tokens=custom_generation_settings.skip_special_tokens,
            stop_token_ids=self.eos_token_id,
            max_tokens=transformers_settings.max_new_tokens,
            # **beam_search_params,
        )
        self._lora_request = lora_request
        self.vllm_engines = vllm_engines
        self._custom_generation_settings = custom_generation_settings
    
    #TODO ?
    def generate_from_batch(self, dataset_name, records, original_records = None):
        return super().generate_from_batch(dataset_name, records, original_records)

    def generate_from_batch_records(
        self,
        dataset_name: str,
        records: list[dict[str, Any]],
        original_records: bool = False,
    ) -> list[ChatInferenceOutput]:
        import os
        #TODO Make sure that records are already splitted between ranks(Assuming micro_rollout_batch_size equal to micro_batch_size)
        input_ids = records['input_ids'].tolist()
        
        rank = torch.distributed.get_rank()
        llm = self.vllm_engines[rank % len(self.vllm_engines)]
        request_outputs = ray.get(llm.generate.remote(sampling_params=self._sampling_params, prompt_token_ids=input_ids, lora_request=self._lora_request))

        outputs = []
        for i, request_output in enumerate(request_outputs):
            answers = []
            for a in request_output.outputs:
                answer_token_ids=torch.tensor(a.token_ids).unsqueeze(0)
                #TODO assuming len(eos_token_id) == 1
                answer_token_ids[:, -1] = self.eos_token_id[0]
                ans_msg = AnswerMessage(
                    id=str(a.index),
                    content=a.text,
                    sequence_score=a.cumulative_logprob,
                    answer_token_ids=answer_token_ids,
                    answer_attention_mask=torch.ones_like(answer_token_ids)
                )

                answers.append(ans_msg)
            if original_records:
                outputs.append(
                    ChatInferenceOutput(
                        input_token_ids=torch.tensor(request_output.prompt_token_ids).unsqueeze(0),
                        input_attention_mask=records['attention_mask'][i, :].unsqueeze(0).cpu(),
                        id=None,
                        dataset_name=dataset_name,
                        messages=None,
                        label=None,
                        answers=answers,
                    )
                )
            else:
                outputs.append(
                    ChatInferenceOutput(
                        input_token_ids=torch.tensor(request_output.prompt_token_ids).unsqueeze(0),
                        dataset_name=dataset_name,
                        answers=answers,
                    )
                )

        return outputs


class ChatGenerator(ChatGeneratorBase[ChatDatasetRecord, ChatInferenceOutput]):
    def generate_from_batch_records(
        self,
        dataset_name: str,
        records_batch: dict[str, torch.Tensor] | BatchEncoding,
        original_records: list[ChatDatasetRecord] | None = None,
        return_logits: bool = None
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

        if return_logits:
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
