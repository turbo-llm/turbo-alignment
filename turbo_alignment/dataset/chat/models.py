from enum import Enum
from typing import Any

from pydantic import BaseModel

from turbo_alignment.dataset.base.models import DatasetRecord


class ChatMessageRole(str, Enum):
    SYSTEM = 'system'
    USER = 'user'
    BOT = 'bot'


class ChatMessage(BaseModel):
    role: ChatMessageRole
    content: str
    disable_loss: bool = False


class ChatDatasetRecord(DatasetRecord):
    messages: list[ChatMessage]
    label: str | None = None
    meta: dict[str, Any] | None = None
