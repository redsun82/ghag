import abc
import dataclasses
import typing

from .types import RefTree
from .expr import RefExpr, Var


def rule(e: RefExpr | Var):
    def decorator(f):
        f.rule_path = e._segments
        return f

    return decorator


@dataclasses.dataclass
class RuleSet(abc.ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._rules = {}
        for f in cls.__dict__.values():
            if callable(f) and hasattr(f, "rule_path"):
                cls._rules.setdefault(len(f.rule_path), []).append((f.rule_path, f))

    @staticmethod
    def _match(lhs: tuple[str, ...], rhs: tuple[str, ...]) -> tuple[str, ...] | None:
        ret = ()
        for l, r in zip(lhs, rhs):
            if r == "*":
                ret += (l,)
            elif l != r:
                return None
        return ret

    @staticmethod
    def _traverse_reftree(
        reftree: RefTree, prefix: tuple[str, ...] = ()
    ) -> typing.Generator[tuple[str, ...], None, None]:
        yield prefix
        for k, rest in reftree.items():
            if k != "*":
                yield from RuleSet._traverse_reftree(rest, prefix + (k,))

    def apply(self, reftree: RefTree, **kwargs: typing.Any) -> bool:
        for path in self._traverse_reftree(reftree):
            for rule, func in self._rules.get(len(path), ()):
                m = self._match(path, rule)
                if m is not None:
                    if not func(self, *m, **kwargs):
                        return False
        return True
