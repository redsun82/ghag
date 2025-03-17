import contextlib
import dataclasses
import typing
from typing import Self, Any
import copy

__all__ = [
    "Expr",
    "Context",
    "MapContext",
    "ContextGroup",
    "Inactive",
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


@dataclasses.dataclass
class _ContextMixin:
    _fields: dict[str, "Expr"] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Expr:
    _value: str | None = None
    _: dataclasses.KW_ONLY
    _op_index: int = 0
    _field_access_error: str | None = ""
    _error: str | typing.Callable[[], str] | None = None

    def __set_name__(self, owner: type, name: str):
        self._value = self._value or name

    @property
    def _attribute_name(self) -> str:
        return f"_f_{self._value}"

    def __get__(self, instance: _ContextMixin | None, owner: type) -> Self:
        if instance is None:
            return self
        assert isinstance(instance, _ContextMixin)
        ret = instance._fields.get(self._value)
        if ret is None:
            ret = copy.deepcopy(self)
            parent = getattr(instance, "_value", None)
            if parent:
                ret._value = f"{parent}.{self._value}"
            instance._fields[self._value] = ret
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
                "e cannot be coerced to bool: did you mean to use `&` for `and` or `|` for `or`?",
            )
        return True


@dataclasses.dataclass
class Context(_ContextMixin, Expr):
    _inactive_error: str | None = None

    def __post_init__(self):
        self._error = self._inactive_error

    def _clear(self):
        self._error = self._inactive_error
        self._fields.clear()

    def _activate(self, field: str | None = None):
        cls = type(self)
        self._error = None
        if field:
            match getattr(cls, field, None):
                case Inactive() as f:
                    f.activate(self)
                case None:
                    raise AttributeError(f"{cls.__name__} has no field `{field}`")

    @property
    def ALL(self) -> Expr:
        return self._emit_error() or Expr(f"{self._value}.*")


@dataclasses.dataclass
class MapContext[T](Context):
    def __init__(
        self, fieldcls: type[T] = Expr, value: str = None, *args: Any, **kwargs: Any
    ):
        super().__init__(value, **kwargs)
        self._fields = {}
        self._fieldcls = fieldcls
        self._args = args
        self._free = False

    def _clear(self):
        super()._clear()
        self._free = False

    def __getattr__(self, item) -> T:
        if item.startswith("_"):
            raise AttributeError(item)
        if self._free:
            self._activate(item)
        try:
            return self._fields[item]
        except KeyError:
            return super().__getattr__(item)

    def _activate(self, field: str | None = None):
        super()._activate()
        if field:
            value = self._fieldcls(f"{self._value}.{field}", *self._args)
            self._fields[field] = value

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


@dataclasses.dataclass
class ContextGroup(_ContextMixin):
    def __post_init__(self):
        # trigger instantiation of all descriptors
        for a in type(self).__dict__:
            _ = getattr(self, a)

    def clear(self):
        for f in self._fields.values():
            f._clear()

    def activate(self):
        for f in self._fields.values():
            f._activate()


class Inactive[T]:
    def __init__(self, cls: type[T] = Expr, *args: Any, **kwargs: Any):
        assert issubclass(cls, Expr)
        self.descriptor = cls(*args, **kwargs)

    def __set_name__(self, owner: type[Context], name: str):
        self.descriptor.__set_name__(owner, name)

    def __get__(self, instance: _ContextMixin, owner: type) -> T:
        if instance is None:
            return self
        assert isinstance(instance, Context)
        if self.descriptor._value not in instance._fields:
            return instance.__getattr__(self.descriptor._value)
        return instance._fields[self.descriptor._value]

    def activate(self, instance: Context):
        self.descriptor.__get__(instance, type(instance))


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
