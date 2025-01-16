from pathlib import Path
from typing import Any, overload

from transformers import PreTrainedTokenizerBase

from turbo_alignment.common.data.io import read_jsonl
from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.dataset.base import AlignmentDataset
from turbo_alignment.dataset.chat import (
    ChatDatasetRecord,
    ChatMessage,
    TrainChatDataset,
)
from turbo_alignment.dataset.pair_preferences.models import PairPreferenceRecord
from turbo_alignment.dataset.registry import PairPreferenceDatasetTypeRegistry
from turbo_alignment.settings.datasets.base import (
    DatasetSourceSettings,
    DatasetStrategy,
)
from turbo_alignment.settings.datasets.pair_preference import (
    PairPreferenceDatasetSettings,
)

logger = get_project_logger()


@PairPreferenceDatasetTypeRegistry.register(DatasetStrategy.TRAIN)
class PairPreferenceDataset(AlignmentDataset[PairPreferenceRecord]):
    def __init__(
        self,
        source: DatasetSourceSettings,
        settings: PairPreferenceDatasetSettings,
        tokenizer: PreTrainedTokenizerBase,
        read: bool = True,
    ):
        self._add_labels = settings.add_labels
        settings.chat_settings.keep_end = True
        self._chat_dataset = TrainChatDataset(
            source=source,
            settings=settings.chat_settings,
            tokenizer=tokenizer,
            read=False,
        )
        super().__init__(source=source, settings=settings, tokenizer=tokenizer)

        if read:
            self._read()

    def convert_records(self, records: list[PairPreferenceRecord]) -> list[dict[str, Any] | None]:
        chosen_chat_records: list[ChatDatasetRecord] = []
        rejected_chat_records: list[ChatDatasetRecord] = []

        for record in records:
            context = [
                ChatMessage(role=message.role, content=message.content, disable_loss=True)
                for message in record.context
            ]

            chosen = ChatMessage(role=record.answer_w.role, content=record.answer_w.content)
            rejected = ChatMessage(role=record.answer_l.role, content=record.answer_l.content)

            chosen_chat_records.append(ChatDatasetRecord(id=record.id, messages=context + [chosen]))
            rejected_chat_records.append(ChatDatasetRecord(id=record.id, messages=context + [rejected]))

        tokenized_chosen_records = self._chat_dataset.convert_records(chosen_chat_records)
        tokenized_rejected_records = self._chat_dataset.convert_records(rejected_chat_records)

        output: list[dict[str, Any] | None] = []
        for record, chosen_record, rejected_record in zip(
            records, tokenized_chosen_records, tokenized_rejected_records
        ):
            if not (chosen_record and rejected_record):
                continue

            ignore_keys = ['precomputed_margin']
            if not self._add_labels:
                ignore_keys.append('labels')

            chosen_tokens = {k: v.squeeze(0) for k, v in chosen_record.items() if k not in ignore_keys}
            rejected_tokens = {k: v.squeeze(0) for k, v in rejected_record.items() if k not in ignore_keys}

            output.append(
                {
                    'id': record.id,
                    'inputs_w': chosen_tokens,
                    'inputs_l': rejected_tokens,
                    'precomputed_margin': record.precomputed_margin,
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
