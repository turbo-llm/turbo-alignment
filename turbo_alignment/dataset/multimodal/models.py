from typing import Annotated, Literal

from pydantic import BaseModel, Field

from turbo_alignment.dataset.base.models import DatasetRecord
from turbo_alignment.dataset.chat.models import ChatMessageRole
from turbo_alignment.settings.modality import Modality


class MultimodalChatMessage(BaseModel):
    role: ChatMessageRole
    type: Modality
    disable_loss: bool = False


class MultimodalTextMessage(MultimodalChatMessage):
    content: str
    type: Literal[Modality.TEXT] = Modality.TEXT


class MultimodalFileMessage(MultimodalChatMessage):
    content: str


class MultimodalImageMessage(MultimodalFileMessage):
    type: Literal[Modality.IMAGE] = Modality.IMAGE


class MultimodalAudioMessage(MultimodalFileMessage):
    type: Literal[Modality.AUDIO] = Modality.AUDIO


class MultimodalDatasetRecord(DatasetRecord):
    messages: list[
        Annotated[MultimodalImageMessage | MultimodalAudioMessage | MultimodalTextMessage, Field(discriminator='type')]
    ]
    label: str | None = None
