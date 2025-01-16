from turbo_alignment.dataset.base.models import DatasetRecord
from turbo_alignment.dataset.chat.models import ChatMessage


class PairPreferenceRecord(DatasetRecord):
    """
    answer_w - preferred answer
    answer_l - other answer
    """

    context: list[ChatMessage]
    answer_w: ChatMessage
    answer_l: ChatMessage
    precomputed_margin: float | None = None
