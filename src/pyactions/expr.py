import enum
import dataclasses
import typing
from typing import Self, Any

from . import element

__all__ = ["Expr"]


class _Op(enum.Enum):
    none = None
    conjunction = "&&"
    disjunction = "||"


@dataclasses.dataclass
class Expr(element.Element):
    _value: str
    _op: str | None = None

    def asdict(self) -> typing.Any:
        return str(self)

    def __str__(self):
        return f"${{{{ {self._value} }}}}"

    def _as_and_operand(self) -> str:
        return f"({self._value})" if self._op == "||" else self._value

    @classmethod
    def _syntax(cls, v: Any, within_and: bool = False) -> str:
        match v:
            case Expr() as e if within_and:
                return e._as_and_operand()
            case Expr() as e:
                return e._value
            case str() as s:
                return f"'{s.replace("'", "''")}'"
            case _:
                return str(v)

    def __and__(self, other: Any) -> Self:
        return Expr(f"{self._as_and_operand()} && {self._syntax(other, True)}", "&&")

    def __rand__(self, other: Any) -> Self:
        return Expr(f"{self._syntax(other, True)} && {self._as_and_operand()}", "&&")

    def __or__(self, other: Any) -> Self:
        return Expr(f"{self._value} || {self._syntax(other)}", "||")

    def __ror__(self, other: Any) -> Self:
        return Expr(f"{self._syntax(other)} || {self._value}", "||")

    def __invert__(self) -> Self:
        operand = f"({self._value})" if self._op else self._value
        return Expr(f"!{operand}")


type Value[T] = Expr | T
