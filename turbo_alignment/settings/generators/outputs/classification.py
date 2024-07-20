from turbo_alignment.dataset.chat.models import ChatMessage
from turbo_alignment.settings.generators.outputs.base import BaseInferenceOutput


class ClassificationInferenceOutput(BaseInferenceOutput):
    id: str
    messages: list[ChatMessage]
    predicted_label: int
