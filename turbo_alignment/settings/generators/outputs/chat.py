from typing import Any

import torch

from turbo_alignment.dataset.chat import ChatMessage
from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel
from turbo_alignment.settings.generators.outputs.base import BaseInferenceOutput


class AnswerMessage(ExtraFieldsNotAllowedBaseModel):
    id: str
    content: str
    answer_token_ids: torch.Tensor
    answer_attention_mask: torch.Tensor
    sequence_score: float | None = None
    logits: torch.Tensor | None = None

    class Config:
        arbitrary_types_allowed = True


class ChatInferenceOutput(BaseInferenceOutput):
    id: str | None = None
    input_token_ids: torch.Tensor | None = None
    input_attention_mask: torch.Tensor | None = None
    answers: list[AnswerMessage]
    messages: list[ChatMessage] | None = None
    label: str | None = None
    meta: dict[str, Any] | None = None

    class Config:
        arbitrary_types_allowed = True


class RagInferenceOutput(ChatInferenceOutput):
    documents: list[str]
    doc_scores: list[float]
