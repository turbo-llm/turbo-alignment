# mypy: ignore-errors
import copy
import inspect
from typing import Any, Callable, Generic, Type, TypeVar

from turbo_alignment.common.registry.params import Params

T = TypeVar('T')


class Lazy(Generic[T]):
    def __init__(
        self,
        constructor: Type[T] | Callable[..., T],
        params: Params | None = None,
        constructor_extras: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        self._constructor = constructor
        self._params = params or Params({})
        self._constructor_extras = constructor_extras or {}
        self._constructor_extras.update(kwargs)

    @property
    def constructor(self) -> Callable[..., T]:
        if inspect.isclass(self._constructor):

            def constructor_to_use(**kwargs):
                return self._constructor.from_params(  # type: ignore[union-attr]
                    copy.deepcopy(self._params),
                    **kwargs,
                )

            return constructor_to_use

        return self._constructor

    def construct(self, **kwargs) -> T:
        contructor_kwargs = {**self._constructor_extras, **kwargs}
        return self.constructor(**contructor_kwargs)
