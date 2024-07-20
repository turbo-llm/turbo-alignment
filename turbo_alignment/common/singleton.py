from collections import defaultdict
from typing import Any, Generic, Type, TypeVar

from pydantic import BaseModel


class Singleton(type):
    _instances: dict[Type, Any] = {}

    def __call__(cls, *args, **kwargs) -> Any:
        if cls not in cls._instances[cls]:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


_SettingsType = TypeVar('_SettingsType', bound=BaseModel)


class ParametrizedSingleton(type, Generic[_SettingsType]):
    _instances: dict[Type, dict[str, Any]] = defaultdict(dict)

    def __call__(cls, settings: _SettingsType) -> Any:
        dumped_settings = settings.json()
        if dumped_settings not in cls._instances[cls]:
            cls._instances[cls][dumped_settings] = super(ParametrizedSingleton, cls).__call__(settings)
        return cls._instances[cls][dumped_settings]
