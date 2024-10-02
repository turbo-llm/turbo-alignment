import torch

from turbo_alignment.dataset.chat import ChatMessage
from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel
from turbo_alignment.settings.generators.outputs.base import BaseInferenceOutput
from typing import Any


class AnswerMessage(ExtraFieldsNotAllowedBaseModel):
    id: str
    content: str
    sequence_score: float | None = None
    answer_token_ids: torch.Tensor | None = None
    logits: torch.Tensor | None = None

    class Config:
        arbitrary_types_allowed = True


class ChatInferenceOutput(BaseInferenceOutput):
    input_token_ids: torch.Tensor | None
    answers: list[AnswerMessage]
    messages: list[ChatMessage] | None
    label: str | None = None
    meta: dict[str, Any] | None = None


class RagInferenceOutput(ChatInferenceOutput):
    documents: list[str]
    doc_scores: list[float]
