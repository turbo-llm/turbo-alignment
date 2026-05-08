from pathlib import Path
from typing import Any, overload

from transformers import PreTrainedTokenizerBase
from typing_extensions import Self

from turbo_alignment.common.data.io import read_jsonl
from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.dataset.base import AlignmentDataset
from turbo_alignment.dataset.chat import (
    ChatDatasetRecord,
    ChatMessage,
)
from turbo_alignment.dataset.chat.chat import SplitChatDataset
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
        seed: int,
        read: bool = True,
    ):
        self._add_labels = settings.add_labels
        settings.chat_settings.keep_end = True
        self._chat_dataset = SplitChatDataset(
            source=source,
            settings=settings.chat_settings,
            tokenizer=tokenizer,
            seed=seed,
            read=False,
        )
        super().__init__(source=source, settings=settings, tokenizer=tokenizer, seed=seed)
        self.settings: PairPreferenceDatasetSettings = settings

        if read:
            self._read()

    def _build_chat_record(self, record: PairPreferenceRecord, answer: ChatMessage) -> ChatDatasetRecord:
        context_messages = [
            ChatMessage(role=msg.role, content=msg.content, disable_loss=True) for msg in record.context
        ]
        return ChatDatasetRecord(id=record.id, messages=context_messages + [answer])

    def convert_records(self, records: list[PairPreferenceRecord]) -> list[dict[str, Any] | None]:
        # Build chat records for chosen and rejected using helper method
        chosen_chat_records = [
            self._build_chat_record(record, ChatMessage(role=record.answer_w.role, content=record.answer_w.content))
            for record in records
        ]
        rejected_chat_records = [
            self._build_chat_record(record, ChatMessage(role=record.answer_l.role, content=record.answer_l.content))
            for record in records
        ]

        tokenized_chosen = self._chat_dataset.convert_records(chosen_chat_records)
        tokenized_rejected = self._chat_dataset.convert_records(rejected_chat_records)

        output: list[dict[str, Any] | None] = []
        for record, chosen_tok, rejected_tok in zip(records, tokenized_chosen, tokenized_rejected):
            if not (chosen_tok and rejected_tok):
                continue

            output.append(
                {
                    'id': record.id,
                    'inputs_context': chosen_tok['context_ids'].squeeze(0),
                    'inputs_chosen': chosen_tok['answer_ids'].squeeze(0),
                    'inputs_rejected': rejected_tok['answer_ids'].squeeze(0),
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

    def get_slice(self, start: int, end: int) -> Self:
        new_instance = self.__class__(
            source=self.source,
            settings=self.settings,
            tokenizer=self.tokenizer,
            seed=self.seed,
            read=False,
        )

        dataset_records = [self[idx] for idx in range(len(self))]

        new_instance.records = self.records[start:end]
        new_instance.original_records_map = {
            record['id']: self.get_original_record_by_id(record['id']) for record in dataset_records
        }

        return new_instance
