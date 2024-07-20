from turbo_alignment.dataset.base.models import DatasetRecord
from turbo_alignment.dataset.chat.models import ChatMessage


class KTODatasetRecord(DatasetRecord):
    context: list[ChatMessage]
    answer: ChatMessage
    is_desirable: bool
