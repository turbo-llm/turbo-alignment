import math
from abc import abstractmethod
from typing import Any, Generic, TypeVar

import torch
from accelerate import Accelerator
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase

from turbo_alignment.dataset.base import BaseDataset
from turbo_alignment.dataset.base.models import DatasetRecord
from turbo_alignment.settings.generators.chat import CustomChatGenerationSettings
from turbo_alignment.settings.generators.outputs.base import BaseInferenceOutput
from turbo_alignment.settings.tf.generation import GeneratorTransformersSettings

InferenceOutputT = TypeVar('InferenceOutputT', bound=BaseInferenceOutput)
DatasetRecordT = TypeVar('DatasetRecordT', bound=DatasetRecord)


class BaseGenerator(Generic[DatasetRecordT, InferenceOutputT]):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        accelerator: Accelerator | None = None,
        batch: int = 1,
    ) -> None:
        if accelerator is not None:
            self._model = accelerator.prepare_model(model, device_placement=True, evaluation_mode=True)
        else:
            self._model = model

        self._tokenizer = tokenizer
        self._accelerator = accelerator
        self._batch = batch

    @abstractmethod
    def generate_from_batch(
        self,
        dataset_name: str,
        records: list[dict[str, Any]],
        original_records: list[DatasetRecordT] | None = None,
    ) -> list[InferenceOutputT]:
        ...

    @property
    def device(self):
        return self._accelerator.device if self._accelerator is not None else self._model.device

    def generate_from_dataset(self, dataset: BaseDataset) -> list[InferenceOutputT]:
        input_records = [dataset[idx] for idx in range(len(dataset))]

        records_batches = [
            input_records[i * self._batch : (i + 1) * self._batch]
            for i in range(math.ceil(len(dataset) / self._batch))
        ]

        original_records_batches = [
            [dataset.get_original_record_by_id(r['id']) for r in batch] for batch in records_batches
        ]

        if self._accelerator is None:
            generations = []
            for i, (records_batch, original_records_batch) in enumerate(
                zip(records_batches, original_records_batches)
            ):
                generations.append(
                    self.generate_from_batch(
                        dataset.source.name,
                        records_batch,
                        original_records_batch,
                    )
                )
        else:
            with self._accelerator.split_between_processes(
                list(zip(records_batches, original_records_batches)), apply_padding=True
            ) as accelerator_records:
                generations = [
                    self.generate_from_batch(
                        dataset.source.name,
                        records_batch,
                        original_records_batch,
                    )
                    for records_batch, original_records_batch in accelerator_records
                ][: len(records_batches)]

        return sum(generations, [])


class ChatGeneratorBase(BaseGenerator, Generic[DatasetRecordT, InferenceOutputT]):
    def __init__(
        self,
        transformers_settings: GeneratorTransformersSettings,
        custom_generation_settings: CustomChatGenerationSettings,
        tokenizer: PreTrainedTokenizerBase,
        return_logits: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(tokenizer=tokenizer, **kwargs)

        self._return_logits = return_logits

        self._transformers_generator_parameters = GenerationConfig(
            bos_token_id=self._tokenizer.bos_token_id,
            **transformers_settings.dict(),
        )

        self._custom_generation_settings = custom_generation_settings

    @abstractmethod
    def generate_from_single_record(
        self,
        dataset_name: str,
        record: dict[str, torch.Tensor],
        original_record: DatasetRecordT | None = None,
    ) -> InferenceOutputT:
        ...

    @abstractmethod
    def generate_from_batch_records(
        self,
        dataset_name: str,
        records_batch: dict[str, torch.Tensor],
        original_records: list[DatasetRecordT] | None = None,
    ) -> list[InferenceOutputT]:
        ...

    def generate_from_batch(
        self, dataset_name: str, records: list[dict[str, Any]], original_records: list[DatasetRecordT] | None = None,
    ) -> list[InferenceOutputT]:
        if self._custom_generation_settings.batch > 1:
            if self._transformers_generator_parameters.num_beams != 1:
                raise ValueError('You can not use batch generation with num_beams != 1')

            self._tokenizer.padding_side = 'left'
            self._tokenizer.pad_token_id = self._tokenizer.pad_token_id

            input_ids = [record['input_ids'].tolist() for record in records]
            attention_mask = [record['attention_mask'].tolist() for record in records]

            max_input_length = max(len(sample) for sample in input_ids)

            # FIXME: type batchencoding, not dict[str, tensor]
            records_batch = self._tokenizer.pad(
                {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                },
                padding='max_length',
                max_length=max_input_length,
                return_tensors='pt',
            )

            return self.generate_from_batch_records(dataset_name, records_batch, original_records)

        return [
            self.generate_from_single_record(dataset_name, record, original_record)
            for record, original_record in zip(records, original_records)
        ]

    @staticmethod
    def _postprocess(
        input_indices: torch.Tensor,
        output_indices: torch.Tensor,
        logits: torch.Tensor | None,
        remove_prompt: bool,
        only_answer_logits: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        processed_logits: torch.Tensor | None = None
        if logits is not None :
            processed_logits = logits.cpu()
            if only_answer_logits:
                processed_logits = logits[:, input_indices.shape[1] :, :].cpu()

        if remove_prompt:
            return output_indices[:, input_indices.shape[1] :].cpu(), processed_logits
        return output_indices.cpu(), logits

    def _decode(self, token_indices: torch.Tensor) -> list[str]:
        return self._tokenizer.batch_decode(
            token_indices, skip_special_tokens=self._custom_generation_settings.skip_special_tokens
        )
