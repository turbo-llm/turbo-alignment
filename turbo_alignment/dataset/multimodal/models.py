from typing import Annotated, Literal

from pydantic import BaseModel, Field

from turbo_alignment.dataset.base.models import DatasetRecord
from turbo_alignment.dataset.chat.models import ChatMessageRole
from turbo_alignment.settings.modality import Modality


class MultimodalMessage(BaseModel):
    role: ChatMessageRole
    content: str
    # modality_object_type: Literal[Modality.IMAGE] = Modality.IMAGE
    modality_object_path: str | None = None
    disable_loss: bool = False


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
        # Annotated[MultimodalImageMessage | MultimodalAudioMessage | MultimodalTextMessage, Field(discriminator='type')]
        MultimodalMessage
    ]
    label: str | None = None
