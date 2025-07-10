import json
import random
from abc import ABC
from itertools import accumulate
from pathlib import Path
from typing import Any, overload

import numpy as np
import numpy.typing as npt
import torch
from transformers import PreTrainedTokenizerBase
from typing_extensions import Self

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
            seed: int,
            read: bool = True,
    ) -> None:
        super().__init__(source=source, settings=settings, tokenizer=tokenizer, seed=seed)
        self.settings: ChatDatasetSettings = settings
        self.cut_generator = random.Random(self.seed)
        self.show_tokenization = False

        if self.settings.dummy_tokens:
            self.dummy_tokens = self.tokenizer(
                self.settings.dummy_tokens,
                truncation=False,
                padding=False,
                add_special_tokens=False
            )['input_ids']

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

        total_length = replicas_cum_len[-1]
        replicas_cum_len_padded = [0] + replicas_cum_len[:-1]

        for left_bound, prev_length in enumerate(replicas_cum_len_padded):
            remaining_length = total_length - prev_length

            if remaining_length <= max_tokens:
                return left_bound, right_bound

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
            right_bound = self.cut_generator.choice(bot_indices) if bot_indices else right_bound

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
        input_ids = input_ids[-(self.settings.max_tokens_count - 1):]  # type: ignore[operator]
        replica_start_token_inds = np.where(input_ids == start_replica_token_id)[0]
        if len(replica_start_token_inds) != 0:
            cut_index = replica_start_token_inds[0]
            input_ids = input_ids[cut_index:]

        labels = labels[-len(input_ids):]
        if self.tokenizer.bos_token_id is not None:
            input_ids = np.concatenate((np.array([self.tokenizer.bos_token_id]), input_ids))
            labels = np.concatenate((np.array([DISABLE_LOSS_LABEL]), labels))

        return input_ids, labels, conversation.get_prompt_repr(left_bound, right_bound)

    def mask_dummy_think_fn(self, chat_tokens, enable_loss):
        dummy_tokens = self.dummy_tokens
        n = len(dummy_tokens)
        for i in range(len(chat_tokens) - n + 1):
            # check for a match of the entire dummy_tokens slice
            if chat_tokens[i: i + n] == dummy_tokens:
                # zero-out that span in enable_loss
                for j in range(n):
                    enable_loss[i + j] = 0
        return enable_loss
    
    def _encode_trl(
            self,
            records: list[ChatDatasetRecord],
            inference: bool,
    ) -> list[dict[str, Any] | None]:
        """
        TRL-style tokenization using HF chat templates.
        This handles whole messages as units rather than token-level truncation.
        """
        result = []
        max_length = self.settings.max_tokens_count

        for record in records:
            tools = record.tools
            if tools is not None and type(tools) == str:
                tools = json.loads(tools)
            if tools is not None:
                tools = [json.loads(tool) if type(tool) == str else tool for tool in tools]
            try:
                # Convert our messages to the format expected by apply_chat_template
                chat_messages = []
                for message in record.messages:
                    # Map our role names to HF expected format

                    role = "system" if message.role == ChatMessageRole.SYSTEM else (
                        "assistant" if message.role == ChatMessageRole.BOT else (
                        "tool" if message.role == ChatMessageRole.TOOL else
                        "documents" if message.role == ChatMessageRole.DOCUMENTS else
                        "user" if  message.role == ChatMessageRole.USER else "unknown"
                    ))
                    chat_messages.append({"role": role, "content": message.content,
                                          "tool_calls": message.tool_calls})

                # For inference, we don't include the last assistant message in the tokenization
                inference_messages = chat_messages
                if inference and chat_messages[-1]["role"] == "assistant":
                    inference_messages = chat_messages[:-1]

                # Apply chat template to the entire conversation
                formatted_chat = self.tokenizer.apply_chat_template(
                    inference_messages,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=inference,
                )


                # Tokenize the chat
                tokenized_chat = self.tokenizer(
                    formatted_chat,
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=False,
                )

                # If the tokenized input is too long, skip this sample
                if max_length is not None and len(tokenized_chat["input_ids"]) > max_length:
                    result.append(None)
                    logger.warning(f"Sample dropped: exceeds max length {max_length}")
                    continue

                # Now we identify and enable loss for assistant responses
                if not inference:
                    loss_mask = np.zeros_like(tokenized_chat["input_ids"], dtype=np.int8)
                    # Process each message to find assistant responses
                    assistant_indices = []
                    for i, message in enumerate(record.messages):
                        if (message.role == ChatMessageRole.BOT and
                                not message.disable_loss and
                                (not self.settings.only_last_replica_loss or i == len(record.messages) - 1)):
                            assistant_indices.append(i)

                    # If no eligible assistant messages, drop this sample
                    if not assistant_indices and self.settings.only_answer_loss:
                        result.append(None)
                        logger.warning("Sample dropped: No assistant messages with enabled loss")
                        continue

                    # For each eligible assistant message, find its tokens in the input_ids
                    # and enable loss for those tokens
                    for idx in assistant_indices:
                        # Apply the template only up to this message to find the start position
                        partial_messages = chat_messages[:idx + 1]
                        partial_format = self.tokenizer.apply_chat_template(
                            partial_messages,
                            tools=tools,
                            tokenize=False,
                            add_generation_prompt=False
                        )

                        # Find the position where this message starts
                        partial_tokens = self.tokenizer(
                            partial_format,
                            truncation=False,
                            padding=False,
                            return_tensors=None,
                            add_special_tokens=False,
                        )["input_ids"]

                        # Exclude the prompt and include only the assistant's response
                        prev_messages = chat_messages[:idx]
                        prev_format = self.tokenizer.apply_chat_template(
                            prev_messages,
                            tools=tools,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        prev_tokens = self.tokenizer(
                            prev_format,
                            truncation=False,
                            padding=False,
                            return_tensors=None,
                            add_special_tokens=False,
                        )["input_ids"]

                        start_idx = len(prev_tokens)
                        end_idx = len(partial_tokens)
                        loss_mask[start_idx:end_idx] = 1

                input_ids_tensor = torch.LongTensor(tokenized_chat["input_ids"])
                attention_mask_tensor = torch.ones(input_ids_tensor.shape, dtype=torch.int64)

                # Optionally mask dummy think tokens

                should_mask = True if self.settings.dummy_tokens else False
                for msg in chat_messages:
                    if msg['role'] in ('user', 'system') and \
                        ('/no_think' in msg['content'] or '/nothink' in msg['content']):
                        should_mask = False
                        break
                
                if should_mask:
                    loss_mask = self.mask_dummy_think_fn(chat_tokens=tokenized_chat["input_ids"],
                                                            enable_loss=loss_mask)

                # Assign labels where mask is 1
                labels = np.where(loss_mask == 1, input_ids_tensor, DISABLE_LOSS_LABEL)

                if self.show_tokenization:
                    filtered_ids = tokenized_chat[labels != DISABLE_LOSS_LABEL]
                    text = self.tokenizer.decode(filtered_ids, skip_special_tokens=False)
                    logger.info(f"Filtered text: {text}")
                    self.show_tokenization = False

                encoded_record = {
                    "input_ids": input_ids_tensor,
                    "attention_mask": attention_mask_tensor,
                    "labels": torch.LongTensor(labels),
                }

                if inference:
                    encoded_record.update({
                        "prompt": formatted_chat,
                        "id": record.id,
                        "messages": record.messages,
                        "meta": record.meta
                    })

                result.append(encoded_record)

            except Exception as ex:
                result.append(None)
                logger.warning(f"Sample dropped: {ex}")

        return result

    def _encode(
            self,
            records: list[ChatDatasetRecord],
            inference: bool,
            random_cut: bool,
    ) -> list[dict[str, Any] | None]:
        # Use TRL tokenization if enabled
        if getattr(self.settings, "use_trl_tokenization", False):
            return self._encode_trl(records, inference)

        # Original tokenization logic
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
                'input_ids': torch.LongTensor(input_ids.astype(np.float32)),
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
            seed: int,
            read: bool = True,
            random_cut: bool = False,
    ) -> None:
        self._random_cut = random_cut

        super().__init__(source=source, settings=settings, tokenizer=tokenizer, read=read, seed=seed)

    def convert_records(self, records: list[ChatDatasetRecord]) -> list[dict[str, Any] | None]:
        return self._encode(records, inference=True, random_cut=self._random_cut)

    def get_slice(self, start: int, end: int) -> Self:
        new_instance = self.__class__(
            source=self.source,
            settings=self.settings,
            tokenizer=self.tokenizer,
            read=False,
            seed=self.seed,
        )

        dataset_records = [self[idx] for idx in range(len(self))]

        new_instance.records = self.records[start:end]
        new_instance.original_records_map = {
            record['id']: self.get_original_record_by_id(record['id']) for record in dataset_records
        }

        return new_instance