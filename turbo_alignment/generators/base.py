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
from turbo_alignment.common.logging import get_project_logger

InferenceOutputT = TypeVar('InferenceOutputT', bound=BaseInferenceOutput)
DatasetRecordT = TypeVar('DatasetRecordT', bound=DatasetRecord)

logger = get_project_logger()


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
    def _generate_from_batch(
        self,
        records: list[dict[str, Any]],
        original_records: list[DatasetRecordT],
        dataset_name: str,
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
                    self._generate_from_batch(
                        records_batch,
                        original_records_batch,
                        dataset.source.name,
                    )
                )
        else:
            with self._accelerator.split_between_processes(
                list(zip(records_batches, original_records_batches)), apply_padding=True
            ) as accelerator_records:
                generations = [
                    self._generate_from_batch(
                        records_batch,
                        original_records_batch,
                        dataset.source.name,
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
    def _generate_from_single_record(
        self,
        record: dict[str, torch.Tensor],
        original_record: DatasetRecordT,
        dataset_name: str,
    ) -> InferenceOutputT:
        ...

    @abstractmethod
    def _generate_from_batch_records(
        self,
        records: list[dict[str, torch.Tensor]],
        original_records: list[DatasetRecordT],
        dataset_name: str,
    ) -> list[InferenceOutputT]:
        ...

    def _generate_from_batch(
        self, records: list[dict[str, Any]], original_records: list[DatasetRecordT], dataset_name: str
    ) -> list[InferenceOutputT]:
        if self._custom_generation_settings.batch > 1:
            if self._transformers_generator_parameters.num_beams != 1:
                raise ValueError('You can not use batch generation with num_beams != 1')

            if self._tokenizer.padding_side == 'right':
                logger.warning(
                    'Changing tokenizer.padding side from "right" to "left".'
                    'This may affect model performance.'
                    'Please verify that `padding_side="left"` is correct for your use case.'
                )
            self._tokenizer.padding_side = 'left'
            self._tokenizer.pad_token_id = self._tokenizer.pad_token_id

            return self._generate_from_batch_records(records, original_records, dataset_name)

        return [
            self._generate_from_single_record(record, original_record, dataset_name)
            for record, original_record in zip(records, original_records)
        ]

    @staticmethod
    def _postprocess(input_indices: torch.Tensor, output_indices: torch.Tensor, remove_prompt: bool) -> torch.Tensor:
        if remove_prompt:
            return output_indices[:, input_indices.shape[1] :].cpu()
        return output_indices.cpu()

    def _decode(self, token_indices: torch.Tensor) -> list[str]:
        return self._tokenizer.batch_decode(
            token_indices, skip_special_tokens=self._custom_generation_settings.skip_special_tokens
        )
