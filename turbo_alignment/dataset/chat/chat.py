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

import gc
from itertools import islice

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

        if read:
            self._read()

    @overload
    def __tokenize(self, inputs: str) -> npt.NDArray[np.int64]: ...

    @overload
    def __tokenize(self, inputs: list[str]) -> np.ndarray: ...

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

        raise ValueError("Can't trim dialogue to fit all requirements")

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

        raise ValueError("Can't trim dialogue to fit all requirements")

    def __truncate(
        self,
        conversation: Conversation,
        replicas_cum_len: list[int],
        inference: bool,
        max_tokens: int | None,
    ) -> tuple[int, int]:
        """
        Truncate the dialogue to satisfy all constraints:
        - cumulative number of tokens is less than max_tokens;
        - keep the end or the start of the dialogue according to settings;
        - remove bot messages from the dialogue end during inference;
        - remove user messages from the dialogue end during training.

        Returns (first_message_index, last_message_index).
        """
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
        # random_cut is used only when inference=True
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
            and right_bound == 1  # if only the system prompt remains after inference
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
        input_ids = input_ids[-(self.settings.max_tokens_count - 1) :]  # type: ignore[operator]
        replica_start_token_inds = np.where(input_ids == start_replica_token_id)[0]
        if len(replica_start_token_inds) != 0:
            cut_index = replica_start_token_inds[0]
            input_ids = input_ids[cut_index:]

        labels = labels[-len(input_ids) :]
        if self.tokenizer.bos_token_id is not None:
            input_ids = np.concatenate((np.array([self.tokenizer.bos_token_id]), input_ids))
            labels = np.concatenate((np.array([DISABLE_LOSS_LABEL]), labels))

        return input_ids, labels, conversation.get_prompt_repr(left_bound, right_bound)

    # logger.info(f'Tokenizing dataset in BATCH-WAY {self.source.name}')
    def _encode(  # type: ignore[override]
        self,
        records: list[ChatDatasetRecord],
        inference: bool,
        random_cut: bool,
    ) -> list[dict[str, Any] | None]:
        """
        Batch tokenization without padding:
        • memory grows proportionally to batch_size, not to the entire dataset;
        • the lengths of replicas are calculated exactly as in the original, which means
          the filters in _truncate_and_merge behave identically,
          so the number of discarded examples matches the "pre‑war" count.
        """
        # ─────────────────────────────────  pre-compute prefixes/suffixes  ─────────────────────────────────
        role_prefix_tokens: dict[ChatMessageRole, np.ndarray] = {
            role: self.__tokenize(
                self.settings.prompt_template.prefix_template.format(
                    role=self.settings.prompt_template.role_tag_mapping[role]
                )
            )[0].astype(
                np.int32
            )  # int32 saves RAM
            for role in ChatMessageRole
        }
        suffix_tokens: np.ndarray = self.__tokenize(self.settings.prompt_template.suffix_template)[0].astype(np.int32)

        batch_size: int = getattr(self.settings, "tokenizer_batch_size", 1024)
        out: list[dict[str, Any] | None] = []

        # ──────────────────────────────────────  process in batches  ──────────────────────────────────────
        for batch_start in range(0, len(records), batch_size):
            batch_records = records[batch_start : batch_start + batch_size]

            # 1) build Conversation objects (they are lightweight)
            conversations = [
                Conversation(
                    system_prompt=self.source.system_prompt,
                    messages=rec.messages,
                    ignore_system_prompt=self.settings.ignore_system_prompt,
                )
                for rec in batch_records
            ]

            # 2) collect all texts of replica in the batch into one list
            texts = [msg.content for conv in conversations for msg in conv.messages]

            # 3) tokenize WITHOUT padding ⇒ returns a "ragged" object‑dtype array
            raw_tokens = self.__tokenize(texts)  # object array
            # 3.1 cast each replica to np.int32
            tokenized_flat = [np.asarray(toks, dtype=np.int32) for toks in raw_tokens]

            # 4) split the flat list back into dialogues
            offset = 0
            for rec, conv in zip(batch_records, conversations):
                num_msgs = len(conv.messages)
                tok_replicas = tokenized_flat[offset : offset + num_msgs]
                offset += num_msgs

                try:
                    input_ids_np, labels_np, prompt = self._truncate_and_merge(
                        conversation=conv,
                        tokenized_replicas=tok_replicas,
                        role_prefix_tokens=role_prefix_tokens,
                        suffix_tokens=suffix_tokens,
                        inference=inference,
                        random_cut=random_cut,
                    )
                except ValueError:  # logic for discarded examples—same as in the original
                    out.append(None)
                    continue

                # 5) form the output dictionary
                encoded: dict[str, Any] = {
                    "input_ids": torch.tensor(input_ids_np, dtype=torch.int32),
                    "labels": torch.tensor(labels_np, dtype=torch.int32),
                    "attention_mask": torch.ones(len(input_ids_np), dtype=torch.int32),
                }
                if inference:
                    encoded.update(
                        {
                            "prompt": prompt,
                            "id": rec.id,
                            "messages": rec.messages,
                            "meta": rec.meta,
                        }
                    )
                out.append(encoded)

            # 6) free memory and proceed to the next batch
            del tokenized_flat, raw_tokens, texts, conversations
            gc.collect()

        return out

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
        
