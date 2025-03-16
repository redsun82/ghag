import contextlib
import dataclasses
import typing
from typing import Self, Any
import copy

from . import element

__all__ = [
    "Expr",
    "Context",
    "MapContext",
    "Field",
    "PotentialField",
    "on_error",
]

_ops = [
    None,
    "[]",
    ".",
    "!",
    " < ",
    " <= ",
    " > ",
    " >= ",
    " == ",
    " != ",
    " && ",
    " || ",
]
_ops = {op: i for i, op in enumerate(_ops)}


class Expr(element.Element):
    _value: str
    _: dataclasses.KW_ONLY
    _op_index: int = 0

    _field_access_error: str | None = ""
    _error: str | typing.Callable[[], str]

    def __set_name__(self, owner: type, name: str):
        self._value = self._value or name

    @property
    def _attribute_name(self) -> str:
        return f"_f_{self._value}"

    def __get__(self, instance: Self | None, owner: type) -> Self:
        if instance is None:
            return self
        # not using `getattr` as `__getattr__` is special
        ret = instance.__dict__.get(self._attribute_name)
        if ret is None:
            ret = copy.deepcopy(self)
            if instance._value:
                ret._value = f"{instance._value}.{self._value}"
            setattr(instance, self._attribute_name, ret)
        return ret

    def _emit_error(self) -> typing.Self | None:
        if self._error is None:
            return None
        if self._error:
            _current_on_error(self._error() if callable(self._error) else self._error)
            self._error = ""
        return self

    def _clear(self):
        pass

    def asdict(self) -> typing.Any:
        return str(self)

    def __str__(self) -> str:
        if self._emit_error():
            return "<error>"
        return f"${{{{ {self._value} }}}}"

    def _as_operand(self, op_index: int) -> str:
        if self._op_index > op_index:
            return f"({self._value})"
        return self._value

    @classmethod
    def _syntax(cls, v: Any, op_index: int | None = None) -> str:
        match v:
            case Expr() as e if op_index is not None:
                return e._as_operand(op_index)
            case Expr() as e:
                return e._value
            case str() as s:
                return f"'{s.replace("'", "''")}'"
            case _:
                return str(v)

    @classmethod
    def _binop(cls, lhs: Any, rhs: Any, op: str) -> Self:
        op_index = _ops[op]
        return Expr(
            f"{cls._syntax(lhs, op_index)}{op}{cls._syntax(rhs, op_index)}",
            _op_index=op_index,
            _field_access_error=None,  # for the moment let's allow attributes on binops
        )

    def __and__(self, other: Any) -> Self:
        return self._emit_error() or self._binop(self, other, " && ")

    def __rand__(self, other: Any) -> Self:
        return self._emit_error() or self._binop(other, self, " && ")

    def __or__(self, other: Any) -> Self:
        return self._emit_error() or self._binop(self, other, " || ")

    def __ror__(self, other: Any) -> Self:
        return self._emit_error() or self._binop(other, self, " || ")

    def __invert__(self) -> Self:
        op_index = _ops["!"]
        return self._emit_error() or Expr(
            f"!{self._as_operand(op_index)}", _op_index=op_index
        )

    def __eq__(self, other: Any) -> Self:
        return self._emit_error() or self._binop(self, other, " == ")

    def __ne__(self, other: Any) -> Self:
        return self._emit_error() or self._binop(self, other, " != ")

    def __le__(self, other) -> Self:
        return self._emit_error() or self._binop(self, other, " <= ")

    def __lt__(self, other) -> Self:
        return self._emit_error() or self._binop(self, other, " < ")

    def __ge__(self, other) -> Self:
        return self._emit_error() or self._binop(self, other, " >= ")

    def __gt__(self, other) -> Self:
        return self._emit_error() or self._binop(self, other, " > ")

    def __getitem__(self, key: Any) -> Self:
        op_index = _ops["[]"]
        return self._emit_error() or Expr(
            f"{self._as_operand(op_index)}[{self._syntax(key, op_index)}]",
            _op_index=op_index,
            _field_access_error=None,
        )

    def __getattr__(self, key: str) -> typing.Self:
        if key.startswith("_"):
            raise AttributeError(key)
        if self._emit_error():
            return self
        if self._field_access_error is not None:
            return ~Expr(
                _error=f"`{key}` not available in `{self._value}`{self._field_access_error}",
            )
        op_index = _ops["."]
        return Expr(f"{self._as_operand(op_index)}.{key}", _op_index=_ops["."])

    def __bool__(self) -> bool:
        if self._emit_error() is None:
            _current_on_error(
                "Expr cannot be coerced to bool: did you mean to use `&` for `and` or `|` for `or`?",
            )
        return True


type Value[T] = Expr | T


def _default_on_error(message: str) -> None:
    raise ValueError(message)


_current_on_error = _default_on_error


@contextlib.contextmanager
def on_error(handler: typing.Callable[[str], typing.Any]):
    global _current_on_error
    _current_on_error = handler
    try:
        yield
    finally:
        _current_on_error = _default_on_error


class Context(Expr):
    _fields: set[str] = dataclasses.field(default_factory=set)
    _inactive_error: str

    def __post_init__(self):
        self._error = self._inactive_error

    def _clear(self):
        self._error = self._inactive_error
        for f in self._fields:
            getattr(self, f)._clear()
            descriptor = getattr(type(self), f, None)
            if descriptor:
                descriptor.clear(self)

    def _activate(self, field: str | None = None):
        cls = type(self)
        self._error = None
        if field:
            match getattr(cls, field, None):
                case None:
                    raise AttributeError(f"{cls.__name__} has no field `{field}`")
                case PotentialField() as f:
                    f.activate(self)
                case _:
                    raise AttributeError(
                        f"{cls.__name__} field `{field}` is not a PotentialField"
                    )

    @property
    def ALL(self) -> Expr:
        return self._emit_error() or Expr(f"{self._value}.*")


class MapContext[T](Context):
    def __init__(
        self, value: str = None, fieldcls: type[T] = Expr, *args: Any, **kwargs: Any
    ):
        super().__init__(value, **kwargs)
        self._fieldcls = fieldcls
        self._args = args
        self._free = False

    def _clear(self):
        super()._clear()
        for f in self._fields:
            delattr(self, f)
        self._fields = set()
        self._free = False

    def __getattr__(self, item) -> T:
        if item.startswith("_"):
            raise AttributeError(item)
        if self._free:
            self._activate(item)
            return getattr(self, item)
        return super().__getattr__(item)

    def _activate(self, field: str | None = None):
        super()._activate()
        if field:
            self._fields.add(field)
            setattr(
                self,
                field,
                self._fieldcls(f"{self._value}.{field}", *self._args),
            )

    def _activate_all(self):
        self._activate()
        self._free = True

    def _has(self, field: str) -> bool:
        return self._free or field in self._fields

    @property
    def ALL(self) -> T:
        return self._emit_error() or self._fieldcls(
            f"{self._value}.*",
            *self._args,
        )


class Field[T]:
    def __init__(self, cls: type[T] = Expr, *args: Any, **kwargs: Any):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def __set_name__(self, owner: type[Context], name: str):
        self.name = name

    @property
    def priv_name(self) -> str:
        return f"_f_{self.name}"

    def __get__(self, instance: Expr, owner: type) -> T:
        if instance is None:
            return self
        instance._fields.add(self.name)
        self.activate(instance)
        return getattr(instance, self.priv_name)

    def activate(self, instance: Expr):
        if self.priv_name not in instance.__dict__:
            setattr(
                instance,
                self.priv_name,
                self.cls(
                    f"{instance._value}.{self.name}",
                    *self.args,
                    **self.kwargs,
                ),
            )

    def clear(self, instance: Expr):
        instance.__dict__.pop(self.priv_name, None)


class PotentialField[T](Field[T]):
    def __get__(self, instance: Expr, owner: type[Context]) -> T:
        if instance is not None and self.priv_name not in instance.__dict__:
            return ~Expr(
                _error=f"`{self.name}` not available in `{instance._value}`{instance._field_access_error}",
            )
        return super().__get__(instance, owner)


def expr_function(name: str, nargs: int = 1) -> typing.Callable[..., Expr]:
    def ret(*args: Expr, **kwargs: Any) -> Expr:
        if len(args) != nargs:
            return ~Expr(
                _error=f"wrong number of arguments to `{name}`, expected {nargs}, got {len(args)}",
            )
        if kwargs:
            return ~Expr(
                _error=f"unexpected keyword arguments to `{name}`, expected {nargs} positional arguments",
            )
        return Expr(f"{name}({', '.join(Expr._syntax(a) for a in args)})")

    return ret
