from pathlib import Path
from typing import Any, overload

from transformers import PreTrainedTokenizerBase

from turbo_alignment.common.data.io import read_jsonl
from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.dataset.base import BaseDataset
from turbo_alignment.dataset.chat import (
    ChatDatasetRecord,
    ChatMessage,
    TrainChatDataset,
)
from turbo_alignment.dataset.pair_preferences import (
    PairPreferenceDataset,
    PairPreferenceRecord,
)
from turbo_alignment.dataset.registry import DDPODatasetTypeRegistry
from turbo_alignment.settings.datasets.base import (
    DatasetSourceSettings,
    DatasetStrategy,
)
from turbo_alignment.settings.datasets.ddpo import DDPOMultiDatasetSettings

logger = get_project_logger()


@DDPODatasetTypeRegistry.register(DatasetStrategy.TRAIN)
class DDPODataset(BaseDataset[PairPreferenceRecord]):
    def __init__(
        self,
        source: DatasetSourceSettings,
        settings: DDPOMultiDatasetSettings,
        chat_tokenizer: PreTrainedTokenizerBase,
        rm_tokenizer: PreTrainedTokenizerBase,
        read: bool = True,
    ) -> None:
        settings.chat_settings.keep_end = True
        self._chat_dataset = TrainChatDataset(
            settings=settings.chat_settings,
            source=source,
            tokenizer=chat_tokenizer,
            read=False,
        )
        self._rm_dataset = PairPreferenceDataset(
            settings=settings.pair_preferences,
            source=source,
            tokenizer=rm_tokenizer,
        )

        super().__init__(
            settings=settings,
            source=source,
        )
        if read:
            self._read()

    def convert_records(self, records: list[PairPreferenceRecord]) -> list[dict[str, Any] | None]:
        chosen_chat_records: list[ChatDatasetRecord] = []
        rejected_chat_records: list[ChatDatasetRecord] = []
        rm_records: list[PairPreferenceRecord] = []

        for record in records:
            context = [
                ChatMessage(role=message.role, content=message.content, disable_loss=True)
                for message in record.context
            ]
            chosen = ChatMessage(role=record.answer_w.role, content=record.answer_w.content)
            rejected = ChatMessage(role=record.answer_l.role, content=record.answer_l.content)

            chosen_chat_records.append(ChatDatasetRecord(id=record.id, messages=context + [chosen]))
            rejected_chat_records.append(ChatDatasetRecord(id=record.id, messages=context + [rejected]))

            rm_records.append(PairPreferenceRecord(id=record.id, context=context, answer_w=chosen, answer_l=rejected))

        tokenized_chosen_records = self._chat_dataset.convert_records(chosen_chat_records)
        tokenized_rejected_records = self._chat_dataset.convert_records(rejected_chat_records)
        tokenized_rm_records = self._rm_dataset.convert_records(rm_records)

        output: list[dict[str, Any] | None] = []
        for record, chosen_record, rejected_record, rm_record in zip(
            records, tokenized_chosen_records, tokenized_rejected_records, tokenized_rm_records
        ):
            if not (chosen_record and rejected_record and rm_record):
                continue

            chosen_tokens = {k: v.squeeze(0) for k, v in chosen_record.items() if k != 'id'}
            rejected_tokens = {k: v.squeeze(0) for k, v in rejected_record.items() if k != 'id'}

            output.append(
                {
                    'sft_inputs_w': chosen_tokens,
                    'sft_inputs_l': rejected_tokens,
                    'rm_inputs_w': rm_record['inputs_w'],
                    'rm_inputs_l': rm_record['inputs_l'],
                }
            )

        return output

    @staticmethod
    @overload
    def _read_records(records: Path) -> list[PairPreferenceRecord]:
        ...

    @staticmethod
    @overload
    def _read_records(records: list[dict]) -> list[PairPreferenceRecord]:
        ...

    @staticmethod
    def _read_records(records) -> list[PairPreferenceRecord]:
        if isinstance(records, Path):
            return [PairPreferenceRecord(**record) for record in read_jsonl(records)]
        if isinstance(records, list):
            return [PairPreferenceRecord(**record) for record in records]
        raise NotImplementedError


def load_ddpo_datasets(
    multi_dataset_settings: DDPOMultiDatasetSettings,
    chat_tokenizer: PreTrainedTokenizerBase,
    rm_tokenizer: PreTrainedTokenizerBase,
) -> list[DDPODataset]:
    logger.info(
        f'Loading dataset {multi_dataset_settings.dataset_type} with settings:\n{multi_dataset_settings.dict()}'
    )

    datasets: list[DDPODataset] = []
    for source in multi_dataset_settings.sources:
        # Determining what type of dataset is in the nested registry by field 'strategy'

        dataset = DDPODataset(
            chat_tokenizer=chat_tokenizer,
            rm_tokenizer=rm_tokenizer,
            source=source,
            settings=multi_dataset_settings,
        )

        datasets.append(dataset)

    return datasets
