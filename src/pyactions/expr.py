import contextlib
import typing
from typing import Self, Any

from . import element

__all__ = [
    "Expr",
    "Context",
    "MapContext",
    "Field",
    "PotentialField",
    "ErrorExpr",
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
    _op_index: int = 0

    _fields: set[str]
    _no_field_error: str

    def _clear(self):
        pass

    def asdict(self) -> typing.Any:
        return str(self)

    def __str__(self) -> str:
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
            f"{cls._syntax(lhs, op_index)}{op}{cls._syntax(rhs, op_index)}", op_index
        )

    def __and__(self, other: Any) -> Self:
        return self._binop(self, other, " && ")

    def __rand__(self, other: Any) -> Self:
        return self._binop(other, self, " && ")

    def __or__(self, other: Any) -> Self:
        return self._binop(self, other, " || ")

    def __ror__(self, other: Any) -> Self:
        return self._binop(other, self, " || ")

    def __invert__(self) -> Self:
        op_index = _ops["!"]
        return Expr(f"!{self._as_operand(op_index)}", op_index)

    def __eq__(self, other: Any) -> Self:
        return self._binop(self, other, " == ")

    def __ne__(self, other: Any) -> Self:
        return self._binop(self, other, " != ")

    def __le__(self, other) -> Self:
        return self._binop(self, other, " <= ")

    def __lt__(self, other) -> Self:
        return self._binop(self, other, " < ")

    def __ge__(self, other) -> Self:
        return self._binop(self, other, " >= ")

    def __gt__(self, other) -> Self:
        return self._binop(self, other, " > ")

    def __getitem__(self, key: Any) -> Self:
        op_index = _ops["[]"]
        return Expr(
            f"{self._as_operand(op_index)}[{self._syntax(key, op_index)}]", op_index
        )

    def __getattr__(self, key: str) -> "Expr | ErrorExpr":
        if self._fields is not None and key not in self._fields:
            return ErrorExpr(
                f"`{key}` not available in `{self._value}`{self._no_field_error or ''}",
                immediate=True,
            )
        op_index = _ops["."]
        return Expr(f"{self._as_operand(op_index)}.{key}")

    def __bool__(self) -> bool:
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


class ErrorExpr(Expr):
    def __init__(
        self, e: str | typing.Callable[[], str] | None = None, immediate: bool = False
    ):
        self.e = e
        if immediate:
            self._emit()

    def _emit(self) -> Self:
        if self.e:
            if callable(self.e):
                self.e = self.e()
            _current_on_error(self.e)
            self.e = None
        return self

    def asdict(self) -> typing.Any:
        return str(self)

    def __str__(self):
        self._emit()
        return "<error>"

    def __call__(self, *args, **kwargs) -> Self:
        return self._emit()

    def __and__(self, other: Any) -> Self:
        return self._emit()

    def __rand__(self, other: Any) -> Self:
        return self._emit()

    def __or__(self, other: Any) -> Self:
        return self._emit()

    def __ror__(self, other: Any) -> Self:
        return self._emit()

    def __invert__(self) -> Self:
        return self._emit()

    def __eq__(self, other: Any) -> Self:
        return self._emit()

    def __ne__(self, other: Any) -> Self:
        return self._emit()

    def __le__(self, other) -> Self:
        return self._emit()

    def __lt__(self, other) -> Self:
        return self._emit()

    def __ge__(self, other) -> Self:
        return self._emit()

    def __gt__(self, other) -> Self:
        return self._emit()

    def __getitem__(self, key: Any) -> Self:
        return self._emit()

    def __getattr__(self, key: Any) -> Self:
        return self._emit()

    def __bool__(self) -> Self:
        return self._emit()


class Context(Expr):
    def __post_init__(self):
        self._fields = set()

    def _clear(self):
        for f in self._fields:
            getattr(self, f)._clear()
            descriptor = getattr(type(self), f, None)
            if descriptor:
                descriptor.clear(self)

    def _activate(self, field: str):
        cls = type(self)
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
        return Expr(f"{self._value}.*")


class MapContext[T](Context):
    def __init__(self, value: str, fieldcls: type[T] = Expr, *args: Any, **kwargs: Any):
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
        if self._free:
            self._activate(item)
            return getattr(self, item)
        return super().__getattr__(item)

    def _activate(self, field: str):
        self._fields.add(field)
        setattr(
            self,
            field,
            self._fieldcls(f"{self._value}.{field}", *self._args, _fields=set()),
        )

    def _activate_all(self):
        self._free = True

    def _has(self, field: str) -> bool:
        return self._free or field in self._fields

    @property
    def ALL(self) -> T:
        return self._fieldcls(f"{self._value}.*", *self._args, fields=set())


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

    def __repr__(self):
        return f"{self.name}@{self.__class__.__name__}({",".join((self.cls.__name__,) + tuple(repr(a) for a in self.args))})"

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
                    _fields=set(),
                    **self.kwargs,
                ),
            )

    def clear(self, instance: Expr):
        instance.__dict__.pop(self.priv_name, None)


class PotentialField[T](Field[T]):
    def __get__(self, instance: Expr, owner: type[Context]) -> T:
        if instance is not None and self.priv_name not in instance.__dict__:
            return ErrorExpr(
                f"`{self.name}` not available in `{instance._value}`", immediate=True
            )
        return super().__get__(instance, owner)


def expr_function(name: str, nargs: int = 1) -> typing.Callable[..., Expr]:
    def ret(*args: Expr, **kwargs: Any) -> Expr:
        if len(args) != nargs:
            return ErrorExpr(
                f"wrong number of arguments to `{name}`, expected {nargs}, got {len(args)}",
                immediate=True,
            )
        if kwargs:
            return ErrorExpr(
                f"unexpected keyword arguments to `{name}`, expected {nargs} positional arguments",
                immediate=True,
            )
        return Expr(f"{name}({', '.join(Expr._syntax(a) for a in args)})")

    return ret
