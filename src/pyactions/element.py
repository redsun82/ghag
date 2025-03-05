import dataclasses
import typing


@dataclasses.dataclass
class Element:
    def asdict(self) -> typing.Any:
        return {
            k.rstrip("_").replace("_", "-"): asobj(v)
            for k, v in (
                (f.name, getattr(self, f.name)) for f in dataclasses.fields(self)
            )
            if v is not None
        }


def asobj(o: typing.Any):
    match o:
        case Element() as e:
            return e.asdict()
        case dict() as d:
            return {k: asobj(v) for k, v in d.items() if v is not None}
        case list() as l:
            return [asobj(x) for x in l]
        case _:
            return o


def element(cls: type) -> type:
    if not issubclass(cls, Element):
        annotations = cls.__annotations__
        cls = type(cls.__name__, (Element,) + cls.__bases__, dict(cls.__dict__))
        cls.__annotations__ = annotations
    for f in cls.__annotations__:
        if not hasattr(cls, f):
            setattr(cls, f, None)
    return dataclasses.dataclass(cls)
