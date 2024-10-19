from typing import Annotated

import torch
from pydantic import Field

from turbo_alignment.dataset.base.models import DatasetRecord
from turbo_alignment.dataset.multimodal.models import (
    MultimodalAudioMessage,
    MultimodalImageMessage,
    MultimodalTextMessage,
)
from turbo_alignment.settings.generators.outputs.base import BaseInferenceOutput
from turbo_alignment.settings.generators.outputs.chat import AnswerMessage


class MultimodalInferenceOutput(BaseInferenceOutput, DatasetRecord):
    messages: list[
        Annotated[MultimodalImageMessage | MultimodalAudioMessage | MultimodalTextMessage, Field(discriminator='type')]
    ]
    label: str | None = None
    answers: list[AnswerMessage]
    input_token_ids: torch.Tensor | None = None

    class Config:
        arbitrary_types_allowed = True
