# mypy: ignore-errors
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    def __reduce__(self) -> str | tuple[Any, ...]:
        return type(self), (self.message,)

    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def __str__(self):
        return self.message
