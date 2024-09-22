from abc import ABC
from pathlib import Path
from typing import Any, overload

import numpy as np
import torch
from safetensors._safetensors_rust import SafetensorError

from turbo_alignment.common.data.io import read_jsonl
from turbo_alignment.common.data.multimodal import BaseModalityReader
from turbo_alignment.common.data.multimodal.registry import ModalityReaderRegistry
from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.common.registry import Params
from turbo_alignment.constants import DISABLE_LOSS_LABEL
from turbo_alignment.dataset.base import AlignmentDataset
from turbo_alignment.dataset.chat.chat import InferenceChatDataset, TrainChatDataset
from turbo_alignment.dataset.chat.models import ChatDatasetRecord, ChatMessage
from turbo_alignment.dataset.multimodal.models import (
    MultimodalDatasetRecord,
    MultimodalFileMessage,
    MultimodalTextMessage,
)
from turbo_alignment.dataset.registry import MultimodalDatasetTypeRegistry
from turbo_alignment.settings.datasets import DatasetStrategy
from turbo_alignment.settings.modality import Modality

VERBOSE_EVERY = 1000

logger = get_project_logger()


class MultimodalDataset(AlignmentDataset[MultimodalDatasetRecord], ABC):
    def __init__(self, tokenizer, source, settings):
        super().__init__(tokenizer=tokenizer, source=source, settings=settings)

        self._n_modality_embeddings = settings.n_modality_embeddings
        self._modality_reader_settings_mapping = settings.modality_reader_settings_mapping
        self._modality_token_mapping = settings.modality_token_mapping
        self._start_modality_token = settings.start_modality_token
        self._end_modality_token = settings.end_modality_token
        self._truncate_top = settings.truncate_top

        self._modality_readers: dict[Modality, BaseModalityReader] = {
            modality: ModalityReaderRegistry.by_name(modality).from_params(
                Params(
                    {
                        'type': self._modality_reader_settings_mapping[modality].reader_type,
                        'reader_path': self._modality_reader_settings_mapping[modality].reader_path,
                    }
                )
            )
            for modality in Modality
            if modality != Modality.TEXT and self._modality_reader_settings_mapping[modality]
        }

    def _tokenize(self, text: str) -> np.ndarray:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors='np',
        )[
            'input_ids'
        ][0]

    def _get_token_id(self, token: str) -> int:
        encoded_token = self._tokenize(token)
        assert len(encoded_token) == 1
        return encoded_token[0]

    @staticmethod
    @overload
    def _read_records(records: Path) -> list[MultimodalDatasetRecord]:
        ...

    @staticmethod
    @overload
    def _read_records(records: list[dict]) -> list[MultimodalDatasetRecord]:
        ...

    @staticmethod
    def _read_records(records) -> list[MultimodalDatasetRecord]:
        if isinstance(records, Path):
            return [MultimodalDatasetRecord(**record) for record in read_jsonl(records)]
        if isinstance(records, list):
            return [MultimodalDatasetRecord(**record) for record in records]
        raise NotImplementedError

    def __get_modality_message(self, modality: Modality) -> str:
        assert modality != Modality.TEXT
        modality_token = self._modality_token_mapping[modality]
        modality_message_span = ''.join(modality_token for _ in range(self._n_modality_embeddings))
        return f'{self._start_modality_token}{modality_message_span}{self._end_modality_token}'

    def _convert_to_chat(self, record: MultimodalDatasetRecord) -> ChatDatasetRecord:
        """
        Обычные текстовые сообщения оставляем без изменений, а мультимодальные сообщения
        преобразуем в <M><modality_tag><modality_tag>...</M>.
        <modality_tag> повторяется n_modality_embeddings раз.
        """

        converted_messages: list[ChatMessage] = []
        for msg in record.messages:
            if isinstance(msg, MultimodalTextMessage):
                converted_messages.append(
                    ChatMessage(role=msg.role, content=msg.content, disable_loss=msg.disable_loss)
                )
            else:
                converted_messages.append(
                    ChatMessage(
                        role=msg.role,
                        content=self.__get_modality_message(msg.type),
                        disable_loss=msg.disable_loss,
                    )
                )

        return ChatDatasetRecord(id=record.id, messages=converted_messages)

    def _read_modalities(
        self, record: MultimodalDatasetRecord, modality_messages_after_truncation: int
    ) -> list[tuple[Modality, torch.Tensor]]:
        modality_messages: list[MultimodalFileMessage] = [
            m for m in record.messages if isinstance(m, MultimodalFileMessage)
        ]

        messages_to_delete = len(modality_messages) - modality_messages_after_truncation

        if self._truncate_top:
            modality_messages = modality_messages[messages_to_delete:]
        else:
            modality_messages = modality_messages[:modality_messages_after_truncation]

        modality_encodings: list[tuple[Modality, torch.Tensor]] = []
        for msg in modality_messages:
            reader = self._modality_readers[msg.type]
            modality_encodings.append((msg.type, reader.read(msg.content)))
        return modality_encodings


@MultimodalDatasetTypeRegistry.register(DatasetStrategy.TRAIN)
class TrainMultimodalDataset(MultimodalDataset):
    def __init__(self, tokenizer, source, settings) -> None:
        """

        :param n_modality_embeddings: сколько токенов выделяем под одно сообщение с нетекстовой модальностью
        :param modality_token_mapping: modality -> token
        :param start_modality_token: начало блока с токенами, которые относятся к нетекстовой модальности
        :param end_modality_token: конец блока с токенами, которые относятся к нетекстовой модальности
        :param truncate_top: обрезаем диалог сверху или снизу, если он не влазит в max_tokens
        """

        super().__init__(tokenizer=tokenizer, source=source, settings=settings)

        self._chat_dataset = TrainChatDataset(tokenizer=tokenizer, source=source, settings=settings, read=False)

        self._read()

    def convert_records(self, records: list[MultimodalDatasetRecord]) -> list[dict[str, Any] | None]:
        chat_records = [self._convert_to_chat(r) for r in records]
        tokenized_chat_records = self._chat_dataset.convert_records(chat_records)

        outputs: list[dict[str, Any] | None] = []

        for i, (record, tokenized_record) in enumerate(zip(records, tokenized_chat_records)):
            if i % VERBOSE_EVERY == 0:
                logger.info(f'{i} records processed of {self.source.name}')

            if tokenized_record is None:
                outputs.append(None)
                continue

            modality_messages_after_truncation = int(
                (tokenized_record['input_ids'] == self._get_token_id(self._start_modality_token)).sum()
            )

            try:
                encoded_modalities = self._read_modalities(record, modality_messages_after_truncation)
            except (OSError, RuntimeError, SafetensorError):
                outputs.append(None)
                continue

            if len(encoded_modalities) != modality_messages_after_truncation:
                outputs.append(None)
                continue

            modality_tokens_mask = torch.isin(
                tokenized_record['input_ids'],
                torch.tensor([self._get_token_id(token) for token in self._modality_token_mapping.values()]),
            )

            tokenized_record['labels'][modality_tokens_mask] = DISABLE_LOSS_LABEL

            outputs.append(
                {
                    **tokenized_record,
                    'modality_inputs': encoded_modalities,
                    'modality_tokens_mask': modality_tokens_mask,
                }
            )

        return outputs


@MultimodalDatasetTypeRegistry.register(DatasetStrategy.INFERENCE)
class InferenceMultimodalDataset(MultimodalDataset):
    def __init__(
        self,
        *args,
        random_cut: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        settings = kwargs['settings']
        settings.random_cut = random_cut
        self._chat_dataset = InferenceChatDataset(
            tokenizer=kwargs['tokenizer'], source=kwargs['source'], settings=settings, read=False
        )
        self._read()

    def convert_records(self, records: list[MultimodalDatasetRecord]) -> list[dict[str, Any] | None]:
        chat_records = [self._convert_to_chat(r) for r in records]
        tokenized_chat_records = self._chat_dataset.convert_records(chat_records)

        outputs: list[dict[str, Any] | None] = []

        for record, tokenized_record in zip(records, tokenized_chat_records):
            if tokenized_record is None:
                outputs.append(None)
                continue

            modality_messages_after_truncation = int(
                (tokenized_record['input_ids'] == self._get_token_id(self._start_modality_token)).sum()
            )

            try:
                encoded_modalities = self._read_modalities(record, modality_messages_after_truncation)
            except (OSError, RuntimeError, SafetensorError):
                outputs.append(None)
                continue

            if len(encoded_modalities) != modality_messages_after_truncation:
                outputs.append(None)
                continue

            modality_tokens_mask = torch.isin(
                tokenized_record['input_ids'],
                torch.tensor([self._get_token_id(token) for token in self._modality_token_mapping.values()]),
            )

            if 'labels' in tokenized_record:
                tokenized_record['labels'][modality_tokens_mask] = DISABLE_LOSS_LABEL

            modality_object_paths = [str(r.content) for r in record.messages if r.type in ('image', 'audio')]

            outputs.append(
                {
                    **tokenized_record,
                    'modality_inputs': encoded_modalities,
                    'modality_tokens_mask': modality_tokens_mask,
                    'modality_object_paths': modality_object_paths,
                }
            )
        return outputs
