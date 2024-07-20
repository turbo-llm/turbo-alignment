import torch

from turbo_alignment.dataset.chat import ChatDatasetRecord
from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel
from turbo_alignment.settings.generators.outputs.base import BaseInferenceOutput


class AnswerMessage(ExtraFieldsNotAllowedBaseModel):
    id: str
    content: str
    sequence_score: float | None = None
    input_token_ids: torch.Tensor | None = None
    answer_token_ids: torch.Tensor | None = None
    logits: torch.Tensor | None = None

    class Config:
        arbitrary_types_allowed = True


class ChatInferenceOutput(BaseInferenceOutput, ChatDatasetRecord):
    answers: list[AnswerMessage]


class RagInferenceOutput(ChatInferenceOutput):
    documents: list[str]
    doc_scores: list[float]
