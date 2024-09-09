import random
from abc import ABC
from enum import Enum
from pathlib import Path
from typing import Any, overload

import loguru
import numpy as np
import numpy.typing as npt
import torch
from jinja2 import Environment
from jinja2.nodes import Assign, Name, Const, List
from transformers import PreTrainedTokenizerBase, BatchEncoding

from turbo_alignment.common.data.io import read_jsonl
from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.constants import DISABLE_LOSS_LABEL
from turbo_alignment.dataset.base import AlignmentDataset
from turbo_alignment.dataset.chat.models import ChatDatasetRecord, ChatMessageRole
from turbo_alignment.dataset.registry import ChatDatasetTypeRegistry
from turbo_alignment.settings.datasets.base import DatasetSourceSettings, DatasetStrategy
from turbo_alignment.settings.datasets.chat import ChatDatasetSettings
from .conversation import Conversation
from turbo_alignment.common.jinja.trackers import AssistantTracker

logger = get_project_logger()


def _get_variable_values(template: str, variable: str) -> list[str]:
    env = Environment(extensions=[AssistantTracker])
    parsed_content = env.parse(template)

    replica_start = None
    for assign_node in parsed_content.find_all(Assign):
        if isinstance(assign_node.target, Name) and assign_node.target.name == variable:
            value_node = assign_node.node
            if isinstance(value_node, Const):
                replica_start = [value_node.value]
            if isinstance(value_node, List):
                replica_start = [item.value for item in value_node.items if isinstance(item, Const)]
    return replica_start


class Tokens:
    def __init__(self, input_ids: npt.NDArray = None, labels: npt.NDArray = None):
        self._input_ids = input_ids if input_ids is not None else np.array([])
        self._labels = labels if labels is not None else np.array([])

    @property
    def input_ids(self):
        return self._input_ids

    @input_ids.setter
    def input_ids(self, input_ids):
        self._input_ids = input_ids

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels):
        self._labels = labels

    def __len__(self):
        return len(self._input_ids)

    def get_suffix(self, from_ind: int) -> "Tokens":
        return Tokens(self._input_ids[from_ind:], self.labels[from_ind:])

    def get_prefix(self, to_ind: int) -> "Tokens":
        return Tokens(self._input_ids[:to_ind], self.labels[:to_ind])

    def append(self, tokens: "Tokens"):
        self._input_ids = np.concatenate((self._input_ids, tokens.input_ids))
        self._labels = np.concatenate((self.labels, tokens.labels))

    def slice(self, from_ind: int, to_ind: int) -> "Tokens":
        return Tokens(self._input_ids[from_ind:to_ind], self._labels[from_ind:to_ind])


def concatenate_tokens(seqs: tuple[Tokens, ...]) -> Tokens:
    result = seqs[0]
    for i in range(1, len(seqs)):
        result.append(seqs[i])
    return result


def get_target_value_indices(input_ids: npt.NDArray[int], target_values: list[int]) -> npt.NDArray[int]:
    return np.where(np.isin(input_ids, target_values))[0]


def enum_to_string_in_dict(conv: list[dict[str, Any]]):
    result = []
    for dictionary in conv:
        for key in dictionary:
            if isinstance(dictionary[key], Enum):
                dictionary[key] = dictionary[key].value
        result.append(dictionary)
    return result


class ChatDataset(AlignmentDataset[ChatDatasetRecord], ABC):
    def __init__(
        self,
        source: DatasetSourceSettings,
        settings: ChatDatasetSettings,
        tokenizer: PreTrainedTokenizerBase,
        read: bool = True,
    ) -> None:
        super().__init__(source=source, settings=settings, tokenizer=tokenizer)
        self.settings: ChatDatasetSettings = settings
        self.prompt_template = settings.prompt_template.read_text()

        replica_start_tokens = _get_variable_values(self.prompt_template, "replica_start")
        replica_end_tokens = _get_variable_values(self.prompt_template, "replica_end")
        if not replica_start_tokens:
            loguru.logger.warning(
                "You should specify tokens that a replica can start with so that the dialogue is splitted correctly."
            )
        self.replica_start_ids = tokenizer.convert_tokens_to_ids(replica_start_tokens)
        self.replica_end_ids = tokenizer.convert_tokens_to_ids(replica_end_tokens)

        if read:
            self._read()

    def _print(self, tokens: Tokens, prefix: str | None = None):
        prefix = prefix + " " if prefix is not None else ""
        loguru.logger.info("{}{}", prefix, [self.tokenizer.batch_decode([id])[0] for id in tokens.input_ids])

    def _encode(
        self,
        records: list[ChatDatasetRecord],
        inference: bool,
        random_cut: bool,
    ) -> list[dict[str, Any] | None]:
        output = []
        for record in records:
            conversation = Conversation(system_prompt=self.source.system_prompt, messages=record.messages)
            unpacked_conversation = [message.dict() for message in conversation.messages]
            unpacked_conversation = enum_to_string_in_dict(unpacked_conversation)

            tokenized_conversation: BatchEncoding = self.tokenizer.apply_chat_template(
                conversation=unpacked_conversation,
                chat_template=self.prompt_template,
                tokenize=True,
                return_assistant_tokens_mask=True,
                return_dict=True,
                add_generation_prompt=inference,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_tensors="np",
                meta=record.meta,
            )
            input_ids = tokenized_conversation["input_ids"][0]

            assistant_masks = tokenized_conversation["assistant_masks"]

            if self.settings.only_last_replica_loss:
                binary_seq_str = ''.join('1' if i else '0' for i in assistant_masks)
                last_replica_start_ind = binary_seq_str.rfind('01')

                result_str = (
                    binary_seq_str[: last_replica_start_ind + 1].replace('1', '0')
                    + binary_seq_str[last_replica_start_ind + 1 :]
                )
                assistant_masks = [i == '1' for i in result_str]

            tokens = Tokens(input_ids, np.where(assistant_masks, input_ids, DISABLE_LOSS_LABEL))
            replica_start_indices = get_target_value_indices(input_ids, self.replica_start_ids)
            replica_end_indices = get_target_value_indices(input_ids, self.replica_end_ids)

            left_replica_ind_inclusive = 0
            right_replica_ind_exclusive = len(replica_start_indices)
            replica_start_indices = np.append(replica_start_indices, replica_end_indices[-1] + 1)

            max_length = self.settings.max_tokens_count

            prefix = Tokens() if replica_start_indices[0] == 0 else tokens.get_prefix(replica_start_indices[0])
            suffix = (
                Tokens()
                if replica_end_indices[-1] == len(input_ids) - 1
                else tokens.get_suffix(replica_end_indices[-1] + 1)
            )

            if conversation.messages[0].role == ChatMessageRole.SYSTEM:
                prefix.append(tokens.slice(replica_start_indices[0], replica_start_indices[1]))
                left_replica_ind_inclusive += 1

            if inference:
                suffix_left_token_ind = replica_start_indices[right_replica_ind_exclusive - 1]
                suffix = concatenate_tokens((tokens.slice(suffix_left_token_ind, replica_end_indices[-1] + 1), suffix))
                right_replica_ind_exclusive -= 1

            max_length = max_length - (len(prefix) + len(suffix))

            while (
                replica_start_indices[right_replica_ind_exclusive] - replica_start_indices[left_replica_ind_inclusive]
                > max_length
            ):
                if self.settings.keep_end:
                    left_replica_ind_inclusive += 1
                else:
                    right_replica_ind_exclusive -= 1

            if random_cut:
                right_replica_ind_exclusive = random.randint(left_replica_ind_inclusive, right_replica_ind_exclusive)

            while right_replica_ind_exclusive > 0 and (
                (inference and conversation.messages[right_replica_ind_exclusive - 1].role == ChatMessageRole.BOT)
                or (
                    not inference
                    and conversation.messages[right_replica_ind_exclusive - 1].role == ChatMessageRole.USER
                )
            ):
                right_replica_ind_exclusive -= 1

            left_token_ind_inclusive = replica_start_indices[left_replica_ind_inclusive]
            right_token_ind_exclusive = replica_start_indices[right_replica_ind_exclusive]
            tokens = concatenate_tokens(
                (prefix, tokens.slice(left_token_ind_inclusive, right_token_ind_exclusive), suffix)
            )

            encoded_record: dict[str, Any] = {
                'input_ids': torch.LongTensor(tokens.input_ids),
                'labels': torch.LongTensor(tokens.labels),
                'attention_mask': torch.ones(tokens.input_ids.shape, dtype=torch.int64),
            }
            if inference:
                encoded_record.update(
                    {
                        'prompt': conversation.get_prompt_repr(
                            left_replica_ind_inclusive, right_replica_ind_exclusive
                        ),
                        'id': record.id,
                        'messages': record.messages,
                        'meta': record.meta,
                    }
                )
            output.append(encoded_record)
        return output

    @staticmethod
    @overload
    def _read_records(records: Path) -> list[ChatDatasetRecord]: ...

    @staticmethod
    @overload
    def _read_records(records: list[dict]) -> list[ChatDatasetRecord]: ...

    @staticmethod
    def _read_records(records) -> list[ChatDatasetRecord]:
        if isinstance(records, Path):
            return [ChatDatasetRecord(**record) for record in read_jsonl(records)]
        if isinstance(records, list):
            return [ChatDatasetRecord(**record) for record in records]
        raise NotImplementedError


@ChatDatasetTypeRegistry.register(DatasetStrategy.TRAIN)
class TrainChatDataset(ChatDataset):
    def convert_records(self, records: list[ChatDatasetRecord]) -> list[dict[str, Any] | None]:
        return self._encode(records, inference=False, random_cut=False)


@ChatDatasetTypeRegistry.register(DatasetStrategy.INFERENCE)
class InferenceChatDataset(ChatDataset):
    def __init__(
        self,
        source: DatasetSourceSettings,
        settings: ChatDatasetSettings,
        tokenizer: PreTrainedTokenizerBase,
        read: bool = True,
        random_cut: bool = False,
    ) -> None:
        self._random_cut = random_cut

        super().__init__(source=source, settings=settings, tokenizer=tokenizer, read=read)

    def convert_records(self, records: list[ChatDatasetRecord]) -> list[dict[str, Any] | None]:
        return self._encode(records, inference=True, random_cut=self._random_cut)
