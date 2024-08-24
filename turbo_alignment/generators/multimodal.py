import torch
from transformers import PreTrainedTokenizerBase

from turbo_alignment.dataset.multimodal.models import MultimodalDatasetRecord
from turbo_alignment.generators.base import ChatGeneratorBase
from turbo_alignment.settings.generators.chat import CustomChatGenerationSettings
from turbo_alignment.settings.generators.outputs.chat import AnswerMessage
from turbo_alignment.settings.generators.outputs.multimodal import (
    MultimodalInferenceOutput,
)
from turbo_alignment.settings.tf.generation import GeneratorTransformersSettings


class MultimodalGenerator(ChatGeneratorBase[MultimodalDatasetRecord, MultimodalInferenceOutput]):
    def __init__(
        self,
        transformers_settings: GeneratorTransformersSettings,
        custom_generation_settings: CustomChatGenerationSettings,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs,
    ) -> None:
        super().__init__(
            transformers_settings=transformers_settings,
            custom_generation_settings=custom_generation_settings,
            tokenizer=tokenizer,
            **kwargs,
        )

    def _generate_from_batch_records(
        self,
        records: list[dict[str, torch.Tensor]],
        original_records: list[MultimodalDatasetRecord],
        dataset_name: str,
    ) -> list[MultimodalInferenceOutput]:
        raise ValueError('You can not use batch generation with multimodаl generator')

    def _generate_from_single_record(
        self,
        record: dict[str, torch.Tensor],
        original_record: MultimodalDatasetRecord,
        dataset_name: str,
    ) -> MultimodalInferenceOutput:
        input_ids = torch.unsqueeze(record['input_ids'], 0).to(self._model.language_model.device)
        attention_mask = torch.unsqueeze(record['attention_mask'], 0).to(self._model.language_model.device)
        modality_inputs = [record['modality_inputs']]
        modality_tokens_mask = record['modality_tokens_mask'].unsqueeze(0).to(self._model.language_model.device)

        inputs_embeds = self._model.convert_inputs_to_embeds(input_ids, modality_inputs, modality_tokens_mask)

        output_indices = self._model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            tokenizer=self._tokenizer,
            generation_config=self._transformers_generator_parameters,
        )

        answers = self._decode(token_indices=output_indices)

        logits: torch.Tensor | None = None
        input_token_ids: torch.Tensor | None = None
        answer_tokens_ids: torch.Tensor | None = None

        if self._return_logits:
            with torch.no_grad():
                logits = self._model(output_indices).logits

            answer_tokens_ids = output_indices
            input_token_ids = input_ids

        return MultimodalInferenceOutput(
            id=original_record.id,
            dataset_name=dataset_name,
            messages=original_record.messages,
            label=original_record.label,
            answers=[
                AnswerMessage(
                    id=str(i),
                    content=a,
                    input_token_ids=input_token_ids,
                    answer_token_ids=answer_tokens_ids,
                    logits=logits,
                )
                for i, a in enumerate(answers)
            ],
        )
