import contextlib
import enum
import dataclasses
import typing
from importlib.resources import open_binary
from typing import Self, Any

from . import element

__all__ = ["Expr", "github"]


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

    fields: tuple[str, ...]

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
        if self.fields is not None and key not in self.fields:
            return ErrorExpr(
                f"`{key}` not available in `{self._value}`", immediate=True
            )
        op_index = _ops["."]
        return Expr(f"{self._as_operand(op_index)}.{key}")

    def __bool__(self) -> "ErrorExpr":
        return ErrorExpr(
            "Expr cannot be coerced to bool: did you mean to use `&` for `and` or `|` for `or`?",
            immediate=True,
        )


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
    @classmethod
    def clear(cls):
        for f in getattr(cls, "_fields", ()):
            f.clear()

    @classmethod
    def activate(cls, field: str):
        match getattr(cls, field, None):
            case None:
                raise AttributeError(f"{cls.__name__} has no field `{field}`")
            case PotentialField() as f:
                f.activate()
            case _:
                raise AttributeError(
                    f"{cls.__name__} field `{field}` is not a PotentialField"
                )


class Field[T]:
    def __init__(self, cls: type[T] = Expr):
        self.cls = cls

    def __set_name__(self, owner: type, name: str):
        if not hasattr(owner, "_fields"):
            owner._fields = []
        owner._fields.append(self)
        self.name = name

    def __get__(self, instance: Expr, owner: type) -> T:
        if instance is None:
            return self
        return self.cls(f"{instance._value}.{self.name}")

    def clear(self):
        getattr(self.cls, "clear", lambda: None)()


class PotentialField[T](Field[T]):
    active = False

    def __get__(self, instance: Expr, owner: type[Context]) -> T:
        if instance is None:
            return self
        if not self.active:
            return ErrorExpr(
                f"`{self.name}` not available in `{instance._value}`", immediate=True
            )
        return super().__get__(instance, owner)

    def activate(self):
        self.active = True

    def clear(self):
        super().clear()
        del self.active


class GithubContext(Context):
    sha = Field()
    ref = Field()
    workflow = Field()
    action = Field()
    actor = Field()
    job = Field()
    run_id = Field()
    run_number = Field()
    event_name = Field()
    event_path = Field()
    action_path = Field()
    workspace = Field()

    class Event(Context):
        action = Field()
        number = Field()
        issue = Field()
        pull_request = PotentialField()
        changes = Field()

    event = Field(Event)


github = GithubContext("github")
