# mypy: ignore-errors
# pylint: skip-file
import collections.abc
import inspect
import logging
from copy import deepcopy
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from turbo_alignment.common.registry.checks import ConfigurationError
from turbo_alignment.common.registry.lazy import Lazy
from turbo_alignment.common.registry.params import Params

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='FromParams')

_NO_DEFAULT = inspect.Parameter.empty


def takes_arg(obj, arg: str) -> bool:
    if inspect.isclass(obj):
        signature = inspect.signature(obj.__init__)
    elif inspect.ismethod(obj) or inspect.isfunction(obj):
        signature = inspect.signature(obj)
    else:
        raise ConfigurationError(f'object {obj} is not callable')
    return arg in signature.parameters


def takes_kwargs(obj) -> bool:
    if inspect.isclass(obj):
        signature = inspect.signature(obj.__init__)
    elif inspect.ismethod(obj) or inspect.isfunction(obj):
        signature = inspect.signature(obj)
    else:
        raise ConfigurationError(f'object {obj} is not callable')
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())  # type: ignore


def can_construct_from_params(type_: Type) -> bool:
    if type_ in [str, int, float, bool]:
        return True
    origin = getattr(type_, '__origin__', None)
    if origin == Lazy:
        return True
    if origin:
        if hasattr(type_, 'from_params'):
            return True
        args = getattr(type_, '__args__')
        return all(can_construct_from_params(arg) for arg in args)

    return hasattr(type_, 'from_params')


def is_base_registrable(cls) -> bool:
    from turbo_alignment.common.registry.registrable import (
        Registrable,  # import here to avoid circular imports
    )

    if not issubclass(cls, Registrable):
        return False
    method_resolution_order = inspect.getmro(cls)[1:]
    for base_class in method_resolution_order:
        if issubclass(base_class, Registrable) and base_class is not Registrable:
            return False
    return True


def remove_optional(annotation: type):
    origin = getattr(annotation, '__origin__', None)
    args = getattr(annotation, '__args__', ())

    if origin == Union:
        return Union[tuple([arg for arg in args if arg != type(None)])]  # noqa: E721
    return annotation


def infer_constructor_params(
    cls: Type[T], constructor: Union[Callable[..., T], Callable[[T], None]] = None
) -> Dict[str, inspect.Parameter]:
    if constructor is None:
        constructor = cls.__init__
    return infer_method_params(cls, constructor)


infer_params = infer_constructor_params  # Legacy name


def infer_method_params(cls: Type[T], method: Callable) -> Dict[str, inspect.Parameter]:
    signature = inspect.signature(method)
    parameters = dict(signature.parameters)

    has_kwargs = False
    var_positional_key = None
    for param in parameters.values():
        if param.kind == param.VAR_KEYWORD:
            has_kwargs = True
        elif param.kind == param.VAR_POSITIONAL:
            var_positional_key = param.name

    if var_positional_key:
        del parameters[var_positional_key]

    if not has_kwargs:
        return parameters

    super_class = None
    for super_class_candidate in cls.mro()[1:]:
        if issubclass(super_class_candidate, FromParams):
            super_class = super_class_candidate
            break
    if super_class:
        super_parameters = infer_params(super_class)
    else:
        super_parameters = {}

    return {**super_parameters, **parameters}


def create_kwargs(constructor: Callable[..., T], cls: Type[T], params: Params, **extras) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}

    parameters = infer_params(cls, constructor)
    accepts_kwargs = False

    for param_name, param in parameters.items():
        if param_name == 'self':
            continue

        if param.kind == param.VAR_KEYWORD:
            accepts_kwargs = True
            continue

        annotation = remove_optional(param.annotation)

        explicitly_set = param_name in params
        # FIXME
        constructed_arg = pop_and_construct_arg(cls.__name__, param_name, annotation, param.default, params, **extras)

        if explicitly_set or constructed_arg is not param.default:
            kwargs[param_name] = constructed_arg

    if accepts_kwargs:
        kwargs.update(params)
    else:
        params.assert_empty(cls.__name__)
    return kwargs


def create_extras(cls: Type[T], extras: Dict[str, Any]) -> Dict[str, Any]:
    subextras: Dict[str, Any] = {}
    if hasattr(cls, 'from_params'):
        from_params_method = cls.from_params
    else:
        from_params_method = cls
    if takes_kwargs(from_params_method):
        subextras = extras
    else:
        subextras = {k: v for k, v in extras.items() if takes_arg(from_params_method, k)}
    return subextras


def construct_arg(
    class_name: str,
    argument_name: str,
    popped_params: Params,
    annotation: Type,
    default: Any,
    **extras,
) -> Any:
    origin = getattr(annotation, '__origin__', None)
    args = getattr(annotation, '__args__', [])

    optional = default != _NO_DEFAULT

    if hasattr(annotation, 'from_params'):
        if popped_params is default:
            return default
        if popped_params is not None:
            subextras = create_extras(annotation, extras)

            if isinstance(popped_params, str):
                popped_params = Params({'type': popped_params})
            elif isinstance(popped_params, dict):
                popped_params = Params(popped_params)
            result = annotation.from_params(params=popped_params, **subextras)

            return result
        if not optional:
            raise ConfigurationError(f'expected key {argument_name} for {class_name}')
        return default

    if annotation in {int, bool}:
        if type(popped_params) in {int, bool}:
            return annotation(popped_params)
        raise TypeError(f'Expected {argument_name} to be a {annotation.__name__}.')
    if annotation == str:
        if isinstance(popped_params, str) or isinstance(popped_params, Path):
            return str(popped_params)  # type: ignore
        raise TypeError(f'Expected {argument_name} to be a string.')
    if annotation == float:
        if type(popped_params) in {int, float}:
            return popped_params
        raise TypeError(f'Expected {argument_name} to be numeric.')

    if (
        origin in {collections.abc.Mapping, Mapping, Dict, dict}
        and len(args) == 2
        and can_construct_from_params(args[-1])
    ):
        value_cls = annotation.__args__[-1]
        value_dict = {}
        if not isinstance(popped_params, Mapping):
            raise TypeError(f'Expected {argument_name} to be a Mapping (probably a dict or a Params object).')

        for key, value_params in popped_params.items():
            value_dict[key] = construct_arg(
                str(value_cls),
                argument_name + '.' + key,
                value_params,
                value_cls,
                _NO_DEFAULT,
                **extras,
            )

        return value_dict

    if origin in (Tuple, tuple) and all(can_construct_from_params(arg) for arg in args):
        value_list = []

        for i, (value_cls, value_params) in enumerate(zip(annotation.__args__, popped_params)):
            value = construct_arg(
                str(value_cls),
                argument_name + f'.{i}',
                value_params,
                value_cls,
                _NO_DEFAULT,
                **extras,
            )
            value_list.append(value)

        return tuple(value_list)

    if origin in (Set, set) and len(args) == 1 and can_construct_from_params(args[0]):
        value_cls = annotation.__args__[0]

        value_set = set()

        for i, value_params in enumerate(popped_params):
            value = construct_arg(
                str(value_cls),
                argument_name + f'.{i}',
                value_params,
                value_cls,
                _NO_DEFAULT,
                **extras,
            )
            value_set.add(value)

        return value_set

    if origin == Union:
        backup_params = deepcopy(popped_params)

        error_chain: Optional[Exception] = None
        for arg_annotation in args:
            try:
                return construct_arg(
                    str(arg_annotation),
                    argument_name,
                    popped_params,
                    arg_annotation,
                    default,
                    **extras,
                )
            except (ValueError, TypeError, ConfigurationError, AttributeError) as e:
                popped_params = deepcopy(backup_params)
                e.args = (f'While constructing an argument of type {arg_annotation}',) + e.args
                e.__cause__ = error_chain
                error_chain = e

        config_error = ConfigurationError(f'Failed to construct argument {argument_name} with type {annotation}.')
        config_error.__cause__ = error_chain
        raise config_error
    if origin == Lazy:
        if popped_params is default:
            return default

        value_cls = args[0]
        subextras = create_extras(value_cls, extras)
        return Lazy(value_cls, params=deepcopy(popped_params), constructor_extras=subextras)  # type: ignore

    if (
        origin in {collections.abc.Iterable, Iterable, List, list}
        and len(args) == 1
        and can_construct_from_params(args[0])
    ):
        value_cls = annotation.__args__[0]

        value_list = []

        for i, value_params in enumerate(popped_params):
            value = construct_arg(
                str(value_cls),
                argument_name + f'.{i}',
                value_params,
                value_cls,
                _NO_DEFAULT,
                **extras,
            )
            value_list.append(value)

        return value_list

    if isinstance(popped_params, Params):
        return popped_params.as_dict()
    return popped_params


class FromParams:
    @classmethod
    def from_params(
        cls: Type[T],
        params: Params,
        constructor_to_call: Callable[..., T] = None,
        constructor_to_inspect: Union[Callable[..., T], Callable[[T], None]] = None,
        **extras,
    ) -> T:
        from turbo_alignment.common.registry.registrable import (
            Registrable,  # import here to avoid circular imports
        )

        logger.debug(
            f"instantiating class {cls} from params {getattr(params, 'params', params)} "
            f'and extras {set(extras.keys())}'
        )

        if params is None:
            return None

        if isinstance(params, str):
            params = Params({'type': params})

        if not isinstance(params, Params):
            raise ConfigurationError(
                'from_params was passed a `params` object that was not a `Params`. This probably '
                'indicates malformed parameters in a configuration file, where something that '
                'should have been a dictionary was actually a list, or something else. '
                f'This happened when constructing an object of type {cls}.'
            )

        registered_subclasses = Registrable._registry.get(cls)

        if is_base_registrable(cls) and registered_subclasses is None:
            raise ConfigurationError(
                'Tried to construct an abstract Registrable base class that has no registered '
                'concrete types. This might mean that you need to use --include-package to get '
                'your concrete classes actually registered.'
            )

        if registered_subclasses is not None and not constructor_to_call:
            # FIXME
            as_registrable = cast(Type[Registrable], cls)
            default_to_first_choice = as_registrable.default_implementation is not None
            # FIXME
            choice = params.pop_choice(
                'type',
                choices=as_registrable.list_available(),
                default_to_first_choice=default_to_first_choice,
            )
            subclass, constructor_name = as_registrable.resolve_class_name(choice)
            if not constructor_name:
                constructor_to_inspect = subclass.__init__
                constructor_to_call = subclass  # type: ignore
            else:
                constructor_to_inspect = cast(Callable[..., T], getattr(subclass, constructor_name))
                constructor_to_call = constructor_to_inspect

            if hasattr(subclass, 'from_params'):
                extras = create_extras(subclass, extras)
                retyped_subclass = cast(Type[T], subclass)
                return retyped_subclass.from_params(
                    params=params,
                    constructor_to_call=constructor_to_call,
                    constructor_to_inspect=constructor_to_inspect,
                    **extras,
                )
            return subclass(**params)  # type: ignore
        if not constructor_to_inspect:
            constructor_to_inspect = cls.__init__
        if not constructor_to_call:
            constructor_to_call = cls

        if constructor_to_inspect == object.__init__:
            kwargs: Dict[str, Any] = {}
            params.assert_empty(cls.__name__)
        else:
            constructor_to_inspect = cast(Callable[..., T], constructor_to_inspect)
            kwargs = create_kwargs(constructor_to_inspect, cls, params, **extras)

        return constructor_to_call(**kwargs)  # type: ignore

    def to_params(self) -> Params:
        """
        Returns a `Params` object that can be used with `.from_params()` to recreate an
        object just like it.

        This relies on `_to_params()`. If you need this in your custom `FromParams` class,
        override `_to_params()`, not this method.
        """

        def replace_object_with_params(o: Any) -> Any:
            if isinstance(o, FromParams):
                return o.to_params()
            if isinstance(o, List):
                return [replace_object_with_params(i) for i in o]
            if isinstance(o, Set):
                return {replace_object_with_params(i) for i in o}
            if isinstance(o, Dict):
                return {key: replace_object_with_params(value) for key, value in o.items()}
            return o

        return Params(replace_object_with_params(self._to_params()))

    def _to_params(self) -> Dict[str, Any]:
        """
        Returns a dictionary of parameters that, when turned into a `Params` object and
        then fed to `.from_params()`, will recreate this object.

        You don't need to implement this all the time. AllenNLP will let you know if you
        need it.
        """
        raise NotImplementedError()
