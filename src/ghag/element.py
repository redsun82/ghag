import dataclasses
import typing

from .expr import Expr, instantiate


@dataclasses.dataclass
class Element:
    _preserve_underscores: typing.ClassVar[bool] = False

    @classmethod
    def _key(cls, key: str) -> str:
        key = key.rstrip("_")
        if not cls._preserve_underscores:
            key = key.replace("_", "-")
        return key

    def asdict(self) -> typing.Any:
        return {
            self._key(k): asobj(v)
            for k, v in (
                (f.name, getattr(self, f.name)) for f in dataclasses.fields(self)
            )
            if v is not None
        }

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for f, a in cls.__annotations__.items():
            # add `None` as default value for all fields not having a default already
            if a is not dataclasses.KW_ONLY and not hasattr(cls, f):
                ty = cls.__annotations__[f]
                cls.__annotations__[f] |= None
                setattr(
                    cls,
                    f,
                    None,
                )
        if dataclasses.KW_ONLY not in cls.__annotations__.values():
            cls.__annotations__ = {"_": dataclasses.KW_ONLY} | cls.__annotations__

        def __repr__(self):
            args = ", ".join(
                f"{f}={v!r}"
                for f, v in (
                    (f.name, getattr(self, f.name)) for f in dataclasses.fields(self)
                )
                if v is not None
            )
            return f"{type(self).__name__}({args})"

        cls.__repr__ = __repr__
        dataclasses.dataclass(cls)


def asobj(o: typing.Any):
    match o:
        case Element() as e:
            return e.asdict()
        case Expr() | str():
            return instantiate(o)
        case dict() as d:
            return {instantiate(k): asobj(v) for k, v in d.items() if v is not None}
        case list() as l:
            return [asobj(x) for x in l]
        case _:
            return o
