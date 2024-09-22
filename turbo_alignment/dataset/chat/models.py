from enum import Enum
from typing import Any

from pydantic import BaseModel, field_validator

from turbo_alignment.dataset.base.models import DatasetRecord


class ChatMessageRole(str, Enum):
    SYSTEM = 'system'
    USER = 'user'
    BOT = 'bot'


class ChatMessage(BaseModel):
    role: ChatMessageRole
    content: str
    disable_loss: bool = False

    @field_validator('role', mode='before')
    def set_bot_role(cls, values: str) -> str:
        if values == 'assistant':
            return ChatMessageRole.BOT
        return values


class ChatDatasetRecord(DatasetRecord):
    messages: list[ChatMessage]
    label: str | None = None
    meta: dict[str, Any] | None = None
