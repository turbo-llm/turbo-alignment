# mypy: ignore-errors
# pylint: skip-file
import importlib
import inspect
import logging
from collections import defaultdict
from typing import Any, Callable, ClassVar, DefaultDict, Type, TypeVar, cast

from turbo_alignment.common.registry.checks import ConfigurationError
from turbo_alignment.common.registry.from_params import FromParams

logger = logging.getLogger(__name__)

_T = TypeVar('_T')
_RegistrableT = TypeVar('_RegistrableT', bound='Registrable')

_SubclassRegistry = dict[str, tuple[type, str | None]]


class Registrable(FromParams):
    _registry: ClassVar[DefaultDict[type, _SubclassRegistry]] = defaultdict(dict)

    default_implementation: str | None = None

    @classmethod
    def register(
        cls, name: str, constructor: str | None = None, exist_ok: bool = False
    ) -> Callable[[Type[_T]], Type[_T]]:
        registry = Registrable._registry[cls]

        def add_subclass_to_registry(subclass: Type[_T]) -> Type[_T]:
            if name in registry:
                if exist_ok:
                    message = (
                        f'{name} has already been registered as {registry[name][0].__name__}, but '
                        f'exist_ok=True, so overwriting with {cls.__name__}'
                    )
                    logger.info(message)
                else:
                    message = (
                        f'Cannot register {name} as {cls.__name__}; '
                        f'name already in use for {registry[name][0].__name__}'
                    )
                    raise ConfigurationError(message)
            registry[name] = (subclass, constructor)
            return subclass

        return add_subclass_to_registry

    @classmethod
    def by_name(cls: Type[_RegistrableT], name):
        logger.debug(f'instantiating registered subclass {name} of {cls}')
        subclass, constructor = cls.resolve_class_name(name)
        if not constructor:
            return cast(Type[_RegistrableT], subclass)
        return cast(Callable[..., _RegistrableT], getattr(subclass, constructor))

    @classmethod
    def resolve_class_name(cls: Type[_RegistrableT], name: str) -> tuple[Type[_RegistrableT], str | None]:
        if name in Registrable._registry[cls]:
            subclass, constructor = Registrable._registry[cls][name]
            return subclass, constructor
        if '.' in name:
            # This might be a fully qualified class name, so we'll try importing its "module"
            # and finding it there.
            parts = name.split('.')
            submodule = '.'.join(parts[:-1])
            class_name = parts[-1]

            try:
                module = importlib.import_module(submodule)
            except ModuleNotFoundError:
                raise ConfigurationError(
                    f'tried to interpret {name} as a path to a class ' f'but unable to import module {submodule}'
                )

            try:
                subclass = getattr(module, class_name)
                constructor = None
                return subclass, constructor
            except AttributeError:
                raise ConfigurationError(
                    f'tried to interpret {name} as a path to a class '
                    f'but unable to find class {class_name} in {submodule}'
                )

        raise ConfigurationError(
            "If your registered class comes from custom code, you'll need to import "
            "the corresponding modules. If you're using AllenNLP from the command-line, "
            "this is done by using the '--include-package' flag, or by specifying your imports "
            "in a '.allennlp_plugins' file. "
            'Alternatively, you can specify your choices '
            """using fully-qualified paths, e.g. {"model": "my_module.models.MyModel"} """
            'in which case they will be automatically imported correctly.'
        )

    @classmethod
    def list_available(cls) -> list[str]:
        keys = list(Registrable._registry[cls].keys())
        default = cls.default_implementation

        if default is None:
            return keys
        if default not in keys:
            raise ConfigurationError(f'Default implementation {default} is not registered')
        return [default] + [k for k in keys if k != default]

    def _to_params(self) -> dict[str, Any]:
        logger.warning(
            f"'{self.__class__.__name__}' does not implement '_to_params`. Will" f" use Registrable's `_to_params`."
        )

        mro = inspect.getmro(self.__class__)[1:]

        registered_name = None
        for parent in mro:
            try:
                registered_classes = self._registry[parent]
            except KeyError:
                continue

            for name, registered_value in registered_classes.items():
                registered_class, _ = registered_value
                if registered_class == self.__class__:
                    registered_name = name
                    break

            if registered_name is not None:
                break

        if registered_name is None:
            raise KeyError(f"'{self.__class__.__name__}' is not registered")

        parameter_dict = {'type': registered_name}

        for parameter in inspect.signature(self.__class__).parameters.values():
            if parameter.default != inspect.Parameter.empty:
                logger.debug(f'Skipping parameter {parameter.name}')
                continue

            if hasattr(self, parameter.name):
                parameter_dict[parameter.name] = getattr(self, parameter.name)
            elif hasattr(self, f'_{parameter.name}'):
                parameter_dict[parameter.name] = getattr(self, f'_{parameter.name}')
            else:
                logger.warning(f'Could not find a value for positional argument {parameter.name}')
                continue

        return parameter_dict
