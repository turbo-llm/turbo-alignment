from collections import defaultdict
from pathlib import Path
from typing import Any, overload

from transformers import PreTrainedTokenizerBase

from turbo_alignment.common.data.io import read_jsonl
from turbo_alignment.dataset.base import AlignmentDataset
from turbo_alignment.dataset.chat import TrainChatDataset
from turbo_alignment.dataset.chat.models import (
    ChatDatasetRecord,
    ChatMessage,
    ChatMessageRole,
)
from turbo_alignment.dataset.registry import SamplingRMDatasetTypeRegistry
from turbo_alignment.dataset.sampling.models import SamplingDatasetRecord
from turbo_alignment.settings.datasets import DatasetStrategy
from turbo_alignment.settings.datasets.base import DatasetSourceSettings
from turbo_alignment.settings.datasets.chat import ChatDatasetSettings


@SamplingRMDatasetTypeRegistry.register(DatasetStrategy.INFERENCE)
class SamplingRMDataset(AlignmentDataset[SamplingDatasetRecord]):
    def __init__(
        self,
        source: DatasetSourceSettings,
        settings: ChatDatasetSettings,
        tokenizer: PreTrainedTokenizerBase,
        read: bool = True,
    ) -> None:
        settings.keep_end = True
        self._chat_dataset = TrainChatDataset(
            settings=settings,
            source=source,
            tokenizer=tokenizer,
            read=False,
        )
        super().__init__(source=source, settings=settings, tokenizer=tokenizer)
        self.settings: ChatDatasetSettings = settings

        if read:
            self._read()

    def convert_records(self, records: list[SamplingDatasetRecord]) -> list[dict[str, Any] | None]:
        chat_records = []

        for record in records:
            prompt_messages = [ChatMessage(role=message.role, content=message.content) for message in record.messages]

            answer_messages = [
                ChatMessage(role=ChatMessageRole.BOT, content=answer.content.removesuffix('</RS>'))
                for answer in record.answers
            ]

            for msg in answer_messages:
                chat_records.append(ChatDatasetRecord(id=record.id, messages=prompt_messages + [msg]))

        tokenized_chats = self._chat_dataset.convert_records(chat_records)

        tokenized_records: dict[str, Any] = defaultdict(dict)
        total_answers_counter = 0
        for record in records:
            tokenized_records[record.id]['id'] = record.id
            tokenized_records[record.id]['rewards'] = record.rewards
            tokenized_records[record.id]['answers'] = {}

            for answer in record.answers:
                if tokenized_chats[total_answers_counter] is not None:
                    tokenized_records[record.id]['answers'][answer.id] = tokenized_chats[total_answers_counter]
                total_answers_counter += 1

        return [tokenized_records[record.id] for record in records if tokenized_records[record.id] is not None]

    @staticmethod
    @overload
    def _read_records(records: Path) -> list[SamplingDatasetRecord]:
        ...

    @staticmethod
    @overload
    def _read_records(records: list[dict]) -> list[SamplingDatasetRecord]:
        ...

    @staticmethod
    def _read_records(records) -> list[SamplingDatasetRecord]:
        if isinstance(records, Path):
            return [SamplingDatasetRecord(**record) for record in read_jsonl(records)]
        if isinstance(records, list):
            return [SamplingDatasetRecord(**record) for record in records]
        raise NotImplementedError
