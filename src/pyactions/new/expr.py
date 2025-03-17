import dataclasses
import abc
import re
import typing
import weakref
import contextlib


class Expr(abc.ABC):
    _precedence: int = 0

    @property
    def _syntax(self) -> str: ...

    def _get_refs(self) -> typing.Generator["RefExpr", None, None]:
        yield from ()

    @staticmethod
    def _instantiate(x: typing.Any) -> typing.Any:
        match x:
            case Expr() as e:
                return str(e).replace("\0", "")
            case str() as s:
                return s.replace("\0", "")
            case dict():
                return {k: Expr._instantiate(v) for k, v in x.items()}
            case list():
                return [Expr._instantiate(i) for i in x]
            case _:
                return x

    @staticmethod
    def _refs(x: typing.Any) -> typing.Generator["RefExpr", None, None]:
        match x:
            case Expr() as e:
                yield from e._get_refs()
            case str() as s:
                for m in re.finditer("\0(.*?)\0", s):
                    yield RefExpr(*m[1].split("."))
            case dict():
                for v in x.values():
                    yield from Expr._refs(v)
            case list():
                for i in x:
                    yield from Expr._refs(i)
            case _:
                return

    def __str__(self) -> str:
        return f"${{{{ {self._syntax} }}}}"

    def _as_operand(self, op_precedence: int) -> str:
        if self._precedence > op_precedence:
            return f"({self._syntax})"
        return self._syntax

    def _operand_from(self, e: "Expr") -> str:
        return e._as_operand(self._precedence)

    @staticmethod
    def _coerce(x: typing.Any) -> "Expr":
        if isinstance(x, Expr):
            return x
        return LiteralExpr(x)

    def __and__(self, other: typing.Any) -> "Expr":
        return BinOpExpr(self, self._coerce(other), "&&")

    def __rand__(self, other: typing.Any) -> "Expr":
        return BinOpExpr(self._coerce(other), self, "&&")

    def __or__(self, other: typing.Any) -> "Expr":
        return BinOpExpr(self, self._coerce(other), "||")

    def __ror__(self, other: typing.Any) -> "Expr":
        return BinOpExpr(self._coerce(other), self, "||")

    def __invert__(self) -> "Expr":
        return NotExpr(self)

    def __eq__(self, other: typing.Any) -> "Expr":
        return BinOpExpr(self, self._coerce(other), "==")

    def __ne__(self, other: typing.Any) -> "Expr":
        return BinOpExpr(self, self._coerce(other), "!=")

    def __le__(self, other) -> "Expr":
        return BinOpExpr(self, self._coerce(other), "<=")

    def __lt__(self, other) -> "Expr":
        return BinOpExpr(self, self._coerce(other), "<")

    def __ge__(self, other) -> "Expr":
        return BinOpExpr(self, self._coerce(other), ">=")

    def __gt__(self, other) -> "Expr":
        return BinOpExpr(self, self._coerce(other), ">")

    def __getitem__(self, key: typing.Any) -> "Expr":
        return ItemExpr(self, self._coerce(key))

    def __getattr__(self, key: str) -> typing.Self:
        if key.startswith("_"):
            raise AttributeError(key)
        return DotExpr(self, key)

    def __bool__(self):
        _current_on_error(
            "expressions cannot be coerced to bool: did you mean to use `&` for `and` or `|` for `or`?",
        )
        return True


instantiate = Expr._instantiate


def refs(x: typing.Any) -> set["RefExpr"]:
    return set(Expr._refs(x))


@dataclasses.dataclass(frozen=True, eq=False, unsafe_hash=True)
class RefExpr(Expr):
    _segments: tuple[str, ...]
    _store: typing.ClassVar[dict[tuple[str, ...], weakref.ReferenceType["RefExpr"]]] = (
        {}
    )

    def __new__(cls, *args: str, **kwargs: typing.Any):
        ref = cls._store.get(args, lambda: None)()
        if ref is not None:
            assert isinstance(
                ref, cls
            ), f"{type(ref).__name__}({", ".join(map(repr, args))}) was created before this {cls.__name__}"
            return ref
        cls._store.pop(args, None)
        instance = super().__new__(cls)
        cls._store[args] = weakref.ref(instance)
        return instance

    def __init__(self, *args: str):
        super().__init__()
        object.__setattr__(self, "_segments", args)

    @property
    def _path(self) -> str:
        return ".".join(self._segments)

    @property
    def _syntax(self) -> str:
        return f"\0{self._path}\0"

    def _get_refs(self) -> typing.Generator["RefExpr", None, None]:
        yield self

    def __getattr__(self, item):
        if item.startswith("_"):
            raise AttributeError(item)
        return RefExpr(*self._segments, item)


_op_precedence = (
    ("[]",),
    ("!",),
    ("<", "<=", ">", ">=", "==", "!="),
    ("&&",),
    ("||",),
)
_op_precedence = {op: i for i, ops in enumerate(_op_precedence) for op in ops}


@dataclasses.dataclass(frozen=True, eq=False)
class LiteralExpr[T](Expr):
    _value: T

    @property
    def _syntax(self) -> str:
        if isinstance(self._value, str):
            return f"'{self._value.replace("'", "''")}'"
        return repr(self._value)


@dataclasses.dataclass(frozen=True, eq=False)
class BinOpExpr(Expr):
    _left: Expr
    _right: Expr
    _op: str

    @property
    def _precedence(self) -> int:
        return _op_precedence[self._op]

    @property
    def _syntax(self) -> str:
        return f"{self._operand_from(self._left)} {self._op} {self._operand_from(self._right)}"

    def _get_refs(self) -> typing.Generator[RefExpr, None, None]:
        yield from self._left._get_refs()
        yield from self._right._get_refs()


@dataclasses.dataclass(frozen=True, eq=False)
class NotExpr(Expr):
    _expr: Expr

    @property
    def _precedence(self) -> int:
        return _op_precedence["!"]

    @property
    def _syntax(self) -> str:
        return f"!{self._operand_from(self._expr)}"

    def _get_refs(self) -> typing.Generator[RefExpr, None, None]:
        yield from self._expr._get_refs()


@dataclasses.dataclass(frozen=True, eq=False)
class ItemExpr(Expr):
    _expr: Expr
    _index: Expr

    @property
    def _precedence(self) -> int:
        return _op_precedence["[]"]

    @property
    def _syntax(self) -> str:
        return f"{self._operand_from(self._expr)}[{self._index._syntax}]"

    def _get_refs(self) -> typing.Generator[RefExpr, None, None]:
        yield from self._expr._get_refs()
        yield from self._index._get_refs()


@dataclasses.dataclass(frozen=True, eq=False)
class DotExpr(Expr):
    _expr: Expr
    _attr: str

    @property
    def _syntax(self) -> str:
        return f"{self._operand_from(self._expr)}.{self._attr}"

    def _get_refs(self) -> typing.Generator[RefExpr, None, None]:
        yield from self._expr._get_refs()


@dataclasses.dataclass(frozen=True, eq=False)
class CallExpr(Expr):
    _function: str
    _args: tuple[Expr, ...]

    def __init__(self, function: str, *args: Expr):
        super().__init__()
        object.__setattr__(self, "_function", function)
        object.__setattr__(self, "_args", args)

    @property
    def _syntax(self) -> str:
        return f"{self._function}({', '.join(a._syntax for a in self._args)})"

    def _get_refs(self) -> typing.Generator[RefExpr, None, None]:
        for a in self._args:
            yield from a._get_refs()


@dataclasses.dataclass
class ErrorExpr(Expr):
    _error: str | typing.Callable[[], str]
    _emitted: bool = False

    def _emit(self) -> typing.Self:
        if not self._emitted:
            if callable(self._error):
                self._error = self._error()
            _current_on_error(self._error)
            self._emitted = True
        return self

    @property
    def _syntax(self) -> str:
        self._emit()
        e: Expr = CallExpr("error", self._coerce(self._error))
        return e._syntax

    def __and__(self, other: typing.Any) -> Expr:
        return self._emit()

    def __rand__(self, other: typing.Any) -> Expr:
        return self._emit()

    def __or__(self, other: typing.Any) -> Expr:
        return self._emit()

    def __ror__(self, other: typing.Any) -> Expr:
        return self._emit()

    def __invert__(self) -> Expr:
        return self._emit()

    def __eq__(self, other: typing.Any) -> Expr:
        return self._emit()

    def __ne__(self, other: typing.Any) -> Expr:
        return self._emit()

    def __le__(self, other) -> Expr:
        return self._emit()

    def __lt__(self, other) -> Expr:
        return self._emit()

    def __ge__(self, other) -> Expr:
        return self._emit()

    def __gt__(self, other) -> Expr:
        return self._emit()

    def __getitem__(self, key: typing.Any) -> Expr:
        return self._emit()

    def __getattr__(self, key: str) -> Expr:
        if key.startswith("_"):
            raise AttributeError(key)
        return self._emit()


type ContextStructure = dict[str, typing.Optional["ContextStructure"]]


class Context(RefExpr):
    def __init__(self, *path: str, structure: ContextStructure | None = None):
        super().__init__(*path)
        object.__setattr__(self, "_child_factory", None)
        if structure:
            for f, d in structure.items():
                child_factory = lambda key: Context(*path, key, structure=d)
                if f == "*":
                    object.__setattr__(self, "_child_factory", child_factory)
                else:
                    object.__setattr__(self, f, child_factory(f))

    def __getattr__(self, name) -> Expr:
        if name.startswith("_"):
            raise AttributeError(name)
        if self._child_factory:
            return self._child_factory(name)
        return ~ErrorExpr(f"`{name}` not available in `{self._path}`")


class Var:
    def __set_name__(self, owner, name):
        owner.fields[name] = self
        self.name = name


def function(name: str, nargs: int = 1) -> typing.Callable[..., Expr]:
    def ret(*args: Expr, **kwargs: typing.Any) -> Expr:
        if kwargs:
            return ~ErrorExpr(
                f"unexpected keyword arguments to `{name}`, expected {nargs} positional arguments",
            )
        if len(args) != nargs:
            return ~ErrorExpr(
                f"wrong number of arguments to `{name}`, expected {nargs}, got {len(args)}"
            )
        return CallExpr(name, *(Expr._coerce(a) for a in args))

    return ret


def _current_on_error(message: str) -> None:
    raise ValueError(message)


@contextlib.contextmanager
def on_error(handler: typing.Callable[[str], typing.Any]):
    global _current_on_error
    old_error = _current_on_error
    _current_on_error = handler
    try:
        yield
    finally:
        _current_on_error = old_error
