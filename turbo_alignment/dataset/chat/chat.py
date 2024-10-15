import random
from abc import ABC
from itertools import accumulate
from pathlib import Path
from typing import Any, overload

import numpy as np
import numpy.typing as npt
import torch
from transformers import PreTrainedTokenizerBase

from turbo_alignment.common.data.io import read_jsonl
from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.constants import DISABLE_LOSS_LABEL
from turbo_alignment.dataset.base import AlignmentDataset
from turbo_alignment.dataset.chat.models import (
    ChatDatasetRecord,
    ChatMessage,
    ChatMessageRole,
)
from turbo_alignment.dataset.registry import ChatDatasetTypeRegistry
from turbo_alignment.settings.datasets.base import (
    DatasetSourceSettings,
    DatasetStrategy,
)
from turbo_alignment.settings.datasets.chat import ChatDatasetSettings

from .conversation import Conversation

logger = get_project_logger()


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

        if read:
            self._read()

    @overload
    def __tokenize(self, inputs: str) -> npt.NDArray[np.int64]:
        ...

    @overload
    def __tokenize(self, inputs: list[str]) -> np.ndarray:
        ...

    def __tokenize(self, inputs):
        return self.tokenizer(
            inputs,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors='np',
        )['input_ids']

    def __keep_end(
        self,
        conversation: Conversation,
        replicas_cum_len: list[int],
        inference: bool,
        max_tokens: int | None,
    ) -> tuple[int, int]:
        if max_tokens is None:
            return 0, len(replicas_cum_len)

        _, right_bound = self.__keep_start(
            conversation=conversation,
            replicas_cum_len=replicas_cum_len,
            inference=inference,
        )

        left_bound = 0
        last_index = replicas_cum_len[-1]
        for i, replica_end_index in enumerate(replicas_cum_len):
            if last_index - replica_end_index <= max_tokens:
                return left_bound, right_bound

            left_bound = i

        raise ValueError('Can\'t trim dialogue to fit all requirements')

    def __keep_start(
        self,
        conversation: Conversation,
        replicas_cum_len: list[int],
        inference: bool,
        max_tokens: int | None = None,
    ) -> tuple[int, int]:
        for i, (message, end_index) in enumerate(zip(conversation.messages[::-1], replicas_cum_len[::-1])):
            if self.settings.only_answer_loss:
                if inference and message.role == ChatMessageRole.BOT:
                    continue
                if not inference and message.role != ChatMessageRole.BOT:
                    continue

            if max_tokens is None or end_index < max_tokens:
                return 0, len(replicas_cum_len) - i

        raise ValueError('Can\'t trim dialogue to fit all requirements')

    def __truncate(
        self,
        conversation: Conversation,
        replicas_cum_len: list[int],
        inference: bool,
        max_tokens: int | None,
    ) -> tuple[int, int]:
        '''
        truncate dialogue to fit all requirements:
        - cumulative tokens num is less than max_tokens
        - keep dialogue end or start according to settings
        - remove bot's messages from dialog end at inference
        - remove user's messages from dialog end at non inference (training)

        returns [first_message_index, last_message_index]
        '''
        if self.settings.keep_end:
            return self.__keep_end(
                conversation=conversation,
                replicas_cum_len=replicas_cum_len,
                max_tokens=max_tokens,
                inference=inference,
            )

        return self.__keep_start(
            conversation=conversation,
            replicas_cum_len=replicas_cum_len,
            max_tokens=max_tokens,
            inference=inference,
        )

    @staticmethod
    def _all_loss_disabled(leftover_messages: list[ChatMessage]) -> bool:
        loss_flags = [message.disable_loss for message in leftover_messages]
        return sum(loss_flags) == len(leftover_messages)

    def _truncate_and_merge(
        self,
        conversation: Conversation,
        tokenized_replicas: list[np.ndarray],
        role_prefix_tokens: dict[ChatMessageRole, np.ndarray],
        suffix_tokens: np.ndarray,
        inference: bool,
        random_cut: bool,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        # random_cut используется только когда inference=true
        assert inference or not random_cut

        bot_prefix_tokens = role_prefix_tokens[ChatMessageRole.BOT]
        max_tokens = (
            self.settings.max_tokens_count - (len(bot_prefix_tokens) if inference else 1)
            if self.settings.max_tokens_count is not None
            else None
        )

        replicas_len = [
            len(r) + len(role_prefix_tokens[m.role]) + len(suffix_tokens)
            for r, m in zip(tokenized_replicas, conversation.messages)
        ]

        left_bound, right_bound = self.__truncate(
            conversation=conversation,
            replicas_cum_len=list(accumulate(replicas_len)),
            max_tokens=max_tokens,
            inference=inference,
        )

        if not inference and right_bound - left_bound < 2:
            raise ValueError('Less than two messages left after truncation')
        if (
            inference
            and left_bound == 0
            and right_bound == 1  # если при инференсе остался только системный промпт
            and conversation.messages[0].role == ChatMessageRole.SYSTEM
        ):
            raise ValueError('Less than two messages left after truncation')
        if not inference and self._all_loss_disabled(conversation.messages[left_bound:right_bound]):
            raise ValueError('No messages with enabled loss were left after truncations')

        if random_cut:
            bot_indices = [
                i
                for i, m in enumerate(conversation.messages)
                if m.role == ChatMessageRole.BOT and left_bound <= i < right_bound
            ]
            right_bound = random.choice(bot_indices) if bot_indices else right_bound

        input_ids = np.array([])
        labels = np.array([])

        truncated_conversation_messages = conversation.messages[left_bound:right_bound]
        truncated_tokenized_replicas = tokenized_replicas[left_bound:right_bound]

        if self.source.system_prompt is not None and self.settings.keep_end and left_bound != 0:
            truncated_conversation_messages = [conversation.messages[0]] + truncated_conversation_messages
            truncated_tokenized_replicas = [truncated_tokenized_replicas[0]] + truncated_tokenized_replicas

        for ind, (message, tokenized_replica) in enumerate(
            zip(
                truncated_conversation_messages,
                truncated_tokenized_replicas,
            )
        ):
            prefix_tokens = role_prefix_tokens[message.role]
            merged_replica = np.concatenate((prefix_tokens, tokenized_replica, suffix_tokens))
            input_ids = np.concatenate((input_ids, merged_replica))

            if (
                (self.settings.only_last_replica_loss and ind != right_bound - left_bound - 1)
                or (self.settings.only_answer_loss and message.role != ChatMessageRole.BOT)
                or message.disable_loss
            ):
                replica_labels = np.full(merged_replica.shape, DISABLE_LOSS_LABEL)
            else:
                replica_labels = np.concatenate(
                    (
                        np.full(prefix_tokens.shape, DISABLE_LOSS_LABEL),
                        tokenized_replica,
                        suffix_tokens,
                    )
                )
            labels = np.concatenate((labels, replica_labels))

        if inference:
            input_ids = np.concatenate((input_ids, bot_prefix_tokens))
            labels = np.concatenate((labels, np.full(bot_prefix_tokens.shape, DISABLE_LOSS_LABEL)))
        else:
            input_ids = np.append(input_ids, self.tokenizer.eos_token_id)
            labels = np.append(labels, DISABLE_LOSS_LABEL)

        # FIXME
        start_replica_token_id = role_prefix_tokens[ChatMessageRole.BOT][0].item()

        # -1 for bos token
        input_ids = input_ids[-(self.settings.max_tokens_count - 1) :]  # type: ignore[operator]
        replica_start_token_inds = np.where(input_ids == start_replica_token_id)[0]
        if len(replica_start_token_inds) != 0:
            cut_index = replica_start_token_inds[0]
            input_ids = input_ids[cut_index:]

        labels = labels[-len(input_ids) :]
        input_ids = np.concatenate((np.array([self.tokenizer.bos_token_id]), input_ids))
        labels = np.concatenate((np.array([DISABLE_LOSS_LABEL]), labels))

        return input_ids, labels, conversation.get_prompt_repr(left_bound, right_bound)

    def _encode(
        self,
        records: list[ChatDatasetRecord],
        inference: bool,
        random_cut: bool,
    ) -> list[dict[str, Any] | None]:
        conversations = [
            Conversation(
                system_prompt=self.source.system_prompt,
                messages=r.messages,
                ignore_system_prompt=self.settings.ignore_system_prompt,
            )
            for r in records
        ]

        logger.info(f'Tokenizing dataset {self.source.name}')
        tokenized_replicas = self.__tokenize([m.content for c in conversations for m in c.messages])

        tokenized_conversations = []
        offset = 0
        for c in conversations:
            tokenized_conversations.append([tokenized_replicas[offset + i] for i in range(len(c.messages))])
            offset += len(c.messages)

        role_prefix_tokens = {
            role: self.__tokenize(
                self.settings.prompt_template.prefix_template.format(
                    role=self.settings.prompt_template.role_tag_mapping[role]
                )
            )[0]
            for role in ChatMessageRole
        }
        # TODO: what if suffix is empty?
        suffix_tokens = self.__tokenize(self.settings.prompt_template.suffix_template)[0]

        logger.info(f'Postprocessing tokenized data in {self.source.name}')
        output: list[dict[str, Any] | None] = []
        for record, conversation, tokenized_replicas in zip(records, conversations, tokenized_conversations):
            try:
                input_ids, labels, prompt = self._truncate_and_merge(
                    conversation=conversation,
                    tokenized_replicas=tokenized_replicas,
                    role_prefix_tokens=role_prefix_tokens,
                    suffix_tokens=suffix_tokens,
                    inference=inference,
                    random_cut=random_cut,
                )

            except ValueError as ex:
                output.append(None)
                logger.warning(f'Sample dropped: {ex}')
                continue

            encoded_record: dict[str, Any] = {
                # 'id': record.id, FIXME: dont work with collators
                'input_ids': torch.LongTensor(input_ids),
                'labels': torch.LongTensor(labels),
                'attention_mask': torch.ones(input_ids.shape, dtype=torch.int64),
            }
            if inference:
                encoded_record.update(
                    {'prompt': prompt, 'id': record.id, 'messages': record.messages, 'meta': record.meta}
                )

            output.append(encoded_record)

        return output

    @staticmethod
    @overload
    def _read_records(records: Path) -> list[ChatDatasetRecord]:
        ...

    @staticmethod
    @overload
    def _read_records(records: list[dict]) -> list[ChatDatasetRecord]:
        ...

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
