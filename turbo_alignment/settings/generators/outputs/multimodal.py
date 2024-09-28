from typing import Annotated

from pydantic import Field

from turbo_alignment.dataset.base.models import DatasetRecord
from turbo_alignment.dataset.multimodal.models import (
    MultimodalAudioMessage,
    MultimodalImageMessage,
    MultimodalMessage,
    MultimodalTextMessage,
)
from turbo_alignment.settings.generators.outputs.base import BaseInferenceOutput
from turbo_alignment.settings.generators.outputs.chat import AnswerMessage


class MultimodalInferenceOutput(BaseInferenceOutput, DatasetRecord):
    messages: list[
        # Annotated[MultimodalImageMessage | MultimodalAudioMessage | MultimodalTextMessage, Field(discriminator='type')]
        MultimodalMessage
    ]
    label: str | None = None
    answers: list[AnswerMessage]
