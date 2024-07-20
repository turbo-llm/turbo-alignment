from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.dataset.chat.models import ChatMessage, ChatMessageRole

logger = get_project_logger()


class Conversation:
    def __init__(
        self,
        system_prompt: str | None,
        messages: list[ChatMessage],
    ):
        self._messages: list[ChatMessage] = []

        if system_prompt:
            self._messages += [ChatMessage(role=ChatMessageRole.SYSTEM, content=system_prompt, disable_loss=True)]

            if messages[0].role == ChatMessageRole.SYSTEM:
                messages = messages[1:]

        self._messages += messages

    @property
    def messages(self) -> list[ChatMessage]:
        return self._messages

    def get_prompt_repr(self, left_bound: int, right_bound: int) -> str:
        prompt: str = ''
        if self._messages[0].role == ChatMessageRole.SYSTEM and left_bound != 0:
            prompt += f'[{self._messages[0].role}: {self._messages[0].content}]'

        for msg in self._messages[left_bound:right_bound]:
            prompt += f'[{msg.role}: {msg.content}]'
        return prompt

    def __len__(self) -> int:
        return len(self._messages)
