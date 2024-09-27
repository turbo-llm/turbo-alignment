import abc

import torch
from accelerate import Accelerator
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from abc import ABC


from allenai_common import Registrable

from turbo_alignment.dataset.chat import ChatDatasetRecord, ChatMessage, ChatMessageRole
from turbo_alignment.dataset.chat.chat import ChatDataset, InferenceChatDataset
from turbo_alignment.settings.datasets.base import DatasetSourceSettings
from turbo_alignment.settings.datasets.chat import ChatDatasetSettings
from turbo_alignment.settings.generators.chat import CustomChatGenerationSettings
from turbo_alignment.settings.online import LLMActorType
from turbo_alignment.generators.chat import ChatGenerator
from turbo_alignment.settings.tf.generation import GeneratorTransformersSettings


class LLMActor(ABC, Registrable):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_tokens_count: int,
        stop_token_id: int,
        temperature: float,
        accelerator: Accelerator | None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_tokens_count: int = max_tokens_count
        self.stop_token_id: int = stop_token_id
        self.temperature: float = temperature
        self.accelerator = accelerator

    @abc.abstractmethod
    def generate_responses(
        self, 
        model: torch.nn.Module | PreTrainedModel,
        queries: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        query_responses,
        attention_mask,
        response_tokens_mask,
        position_ids,
        rewards,
        meta
        """
        
        raise NotImplementedError


@LLMActor.register(LLMActorType.LOCAL_TRANSFORMERS)
class LocalTransformersLLMActor(LLMActor):

    def generate_responses(
        self, 
        model: torch.nn.Module | PreTrainedModel, 
        queries: torch.Tensor, 
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        generator_transformers_settings = GeneratorTransformersSettings(
            temperature=self.temperature,
            max_length=self.max_tokens_count,
        )
        generator_custom_settings = CustomChatGenerationSettings(batch=1)

        generator = ChatGenerator(
            model=model,
            tokenizer=self.tokenizer,
            transformers_settings=generator_transformers_settings,
            custom_generation_settings=generator_custom_settings,
            accelerator=self.accelerator,
            return_logits=True,
        )

        chat_dataset_records = [
            ChatDatasetRecord(messages=[ChatMessage(role=ChatMessageRole.USER, content=q)], id=str(id))
            for id, q in enumerate(queries)
        ]

        dataset = InferenceChatDataset(
            source=DatasetSourceSettings(name="prompts", records_data=[]),
            settings=ChatDatasetSettings(),
            tokenizer=self.tokenizer
        )

        generations = generator.generate_from_dataset(dataset)

