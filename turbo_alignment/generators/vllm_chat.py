from typing import Any

from transformers import PreTrainedTokenizerBase
from vllm import LLM, SamplingParams

from turbo_alignment.dataset.chat import ChatDatasetRecord
from turbo_alignment.generators.base import BaseGenerator
from turbo_alignment.settings.generators.chat import CustomChatGenerationSettings
from turbo_alignment.settings.generators.outputs.chat import (
    AnswerMessage,
    ChatInferenceOutput,
)
from turbo_alignment.settings.tf.generation import GeneratorTransformersSettings


class VLLMChatGenerator(BaseGenerator[ChatDatasetRecord, ChatInferenceOutput]):
    def __init__(
        self,
        transformers_settings: GeneratorTransformersSettings,
        custom_generation_settings: CustomChatGenerationSettings,
        model: LLM,
        tokenizer: PreTrainedTokenizerBase,
        batch: int,
    ):
        model.set_tokenizer(tokenizer)
        super().__init__(model, tokenizer, batch=batch)

        if isinstance(transformers_settings.stop_strings, list):
            raise ValueError('You should use only 1 eos token with VLLM')

        eos_token_id: list[int] = self._tokenizer.encode(transformers_settings.stop_strings, add_special_tokens=False)

        beam_search_params = {'best_of': transformers_settings.num_return_sequences, 'use_beam_search': False}
        if transformers_settings.num_beams > 1:
            beam_search_params['use_beam_search'] = True
            beam_search_params['best_of'] = transformers_settings.num_beams

        self._sampling_params = SamplingParams(
            n=transformers_settings.num_return_sequences,
            repetition_penalty=transformers_settings.repetition_penalty,
            temperature=transformers_settings.temperature,
            top_p=transformers_settings.top_p,
            top_k=transformers_settings.top_k,
            skip_special_tokens=custom_generation_settings.skip_special_tokens,
            stop_token_ids=eos_token_id,
            max_tokens=transformers_settings.max_new_tokens,
            **beam_search_params,  # type: ignore[arg-type]
        )

    def _generate_from_batch(
        self, records: list[dict[str, Any]], original_records: list[ChatDatasetRecord], dataset_name: str
    ) -> list[ChatInferenceOutput]:
        input_ids = [record['input_ids'].tolist() for record in records]
        prompts = self._tokenizer.batch_decode(sequences=input_ids, skip_special_tokens=False)
        request_outputs = self._model.generate(prompts, self._sampling_params)

        outputs = []
        for i, request_output in enumerate(request_outputs):
            original_record = original_records[i]
            outputs.append(
                ChatInferenceOutput(
                    id=original_record.id,
                    dataset_name=dataset_name,
                    messages=original_record.messages,
                    label=original_record.label,
                    answers=[
                        AnswerMessage(id=str(a.index), content=a.text, sequence_score=a.cumulative_logprob)
                        for a in request_output.outputs
                    ],
                )
            )
        return outputs
