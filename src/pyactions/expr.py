import enum
import dataclasses
import typing
from importlib.resources import open_binary
from typing import Self, Any

from . import element

__all__ = ["Expr"]


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
            return ErrorExpr(f"`{key}` not available in `{self._value}`")
        op_index = _ops["."]
        return Expr(f"{self._as_operand(op_index)}.{key}")

    def __bool__(self) -> "ErrorExpr":
        return ErrorExpr(
            "Expr cannot be coerced to bool: did you mean to use `&` for `and` or `|` for `or`?"
        )


type Value[T] = Expr | T


def on_error(message: str):
    raise Exception(message)


class ErrorExpr(Expr):
    def __init__(self, e: str | None = None):
        if e:
            on_error(e)

    def asdict(self) -> typing.Any:
        return str(self)

    def __str__(self):
        return "<error>"

    def __call__(self, *args, **kwargs) -> Self:
        return self

    def __and__(self, other: Any) -> Self:
        return self

    def __rand__(self, other: Any) -> Self:
        return self

    def __or__(self, other: Any) -> Self:
        return self

    def __ror__(self, other: Any) -> Self:
        return self

    def __invert__(self) -> Self:
        return self

    def __eq__(self, other: Any) -> Self:
        return self

    def __ne__(self, other: Any) -> Self:
        return self

    def __le__(self, other) -> Self:
        return self

    def __lt__(self, other) -> Self:
        return self

    def __ge__(self, other) -> Self:
        return self

    def __gt__(self, other) -> Self:
        return self

    def __getitem__(self, key: Any) -> Self:
        return self

    def __getattr__(self, key: Any) -> Self:
        return self

    def __bool__(self) -> Self:
        return self
