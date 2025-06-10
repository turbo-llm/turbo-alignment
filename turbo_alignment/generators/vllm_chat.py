from typing import Any

import torch
from transformers import PreTrainedTokenizerBase
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from turbo_alignment.dataset.chat import ChatDatasetRecord
from turbo_alignment.generators.base import BaseGenerator
from turbo_alignment.settings.generators.chat import CustomChatGenerationSettings
from turbo_alignment.settings.generators.outputs.chat import (
    AnswerMessage,
    ChatInferenceOutput,
)
from turbo_alignment.settings.tf.generation import VLLMGeneratorSettings


class TokenSuppressionLogitsProcessor:
    def __init__(self, available_tokens: list[int]):
        self.available_tokens = available_tokens

    def __call__(self, generated_tokens: list[int], logits: torch.Tensor) -> torch.Tensor:
        suppress_mask = torch.full_like(logits, float('-inf'))
        suppress_mask[self.available_tokens] = 0

        return logits + suppress_mask


class VLLMChatGenerator(BaseGenerator[ChatDatasetRecord, ChatInferenceOutput]):
    def __init__(
        self,
        generator_settings: VLLMGeneratorSettings,
        custom_generation_settings: CustomChatGenerationSettings,
        model: LLM,
        tokenizer: PreTrainedTokenizerBase,
        batch: int,
        return_logits: bool = False,
        lora_request: LoRARequest | None = None,
    ):
        model.set_tokenizer(tokenizer)  # type: ignore[arg-type]
        super().__init__(model, tokenizer, batch=batch)  # type: ignore[arg-type]

        if isinstance(generator_settings.stop_strings, list):
            raise ValueError('You should use only 1 eos token with VLLM')

        eos_token_id: list[int] = self._tokenizer.encode(generator_settings.stop_strings, add_special_tokens=False)

        beam_search_params: dict[str, Any] = {
            'best_of': generator_settings.best_of if generator_settings.best_of else generator_settings.n,
        }

        sampling_params = generator_settings.dict(
            exclude={'best_of', 'stop_strings', 'filter_token_ids'}
        )
        if generator_settings.filter_token_ids:
            sampling_params['logits_processors'] = [
                TokenSuppressionLogitsProcessor(generator_settings.filter_token_ids)
            ]

        self._sampling_params = SamplingParams(
            **sampling_params,
            skip_special_tokens=custom_generation_settings.skip_special_tokens,
            stop_token_ids=eos_token_id,
            **beam_search_params,
        )
        self._lora_request = lora_request

        self._return_logits = return_logits

    def _generate_from_batch(
        self, records: list[dict[str, Any]], original_records: list[ChatDatasetRecord], dataset_name: str
    ) -> list[ChatInferenceOutput]:
        input_ids = [record['input_ids'].tolist() for record in records]
        request_outputs = self._model.generate(
            prompts=None,
            prompt_token_ids=input_ids,
            sampling_params=self._sampling_params,
            lora_request=self._lora_request,
        )

        outputs = []
        for i, request_output in enumerate(request_outputs):
            original_record = original_records[i]
            answers = []
            for a in request_output.outputs:
                ans_msg = AnswerMessage(
                    id=str(a.index), content=a.text, sequence_score=a.cumulative_logprob, logprobs=a.logprobs
                )
                if self._return_logits:
                    ans_msg.input_token_ids = torch.tensor(request_output.prompt_token_ids).unsqueeze(0)
                    ans_msg.answer_token_ids = torch.tensor(a.token_ids).unsqueeze(0)

                answers.append(ans_msg)

            outputs.append(
                ChatInferenceOutput(
                    id=original_record.id,
                    dataset_name=dataset_name,
                    messages=original_record.messages,
                    label=original_record.label,
                    answers=answers,
                )
            )
        return outputs
