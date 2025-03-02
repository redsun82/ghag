import typing
from dataclasses import dataclass, fields
import dataclasses
import threading

from .element import Element
from .workflow import *


@dataclass
class _Context(threading.local):
    current: Workflow | Job | None = None


_ctx = _Context()

def _get_field(field: str, instance: typing.Any = None) -> dataclasses.Field:
    return next(f for f in fields(instance or _ctx.current) if f.name == field)


def _update_field(field: str, value = None, **kwargs):
    assert bool(value) + bool(kwargs) <= 1
    f = _get_field(field)
    current_value = getattr(_ctx.current, field)
    if value is not None:
        setattr(_ctx.current, field, value)
    elif f.type is dict or f.type.__origin__ is dict and current_value is not None:
        current_value |= kwargs
    else:
        setattr(_ctx.current, field, f.type(**kwargs))

def _update_subfield(field: str, subfield: str, value=None, **kwargs):
    f = _get_field(field)
    if getattr(_ctx.current, field) is None:
        _update_field(field)
    instance = getattr(_ctx.current, field)
    sf = _get_field(subfield, instance)
    setattr(instance, subfield, value or sf.type(**kwargs))


class _OnUpdater:
    def pull_request(self, **kwargs) -> typing.Self:
        _update_subfield("on", "pull_request", **kwargs)
        return self

    def workflow_dispatch(self, **kwargs) -> typing.Self:
        _update_subfield("on", "workflow_dispatch", **kwargs)
        return self


@dataclass
class WorkflowInfo:
    id: str
    spec: typing.Callable[[], None]

    def instantiate(self) -> Workflow:
        ret = _ctx.current = Workflow()
        self.spec()
        _ctx.current = None
        return ret


def workflow(func=None, *, id=None) -> WorkflowInfo:
    if func is None:
        return lambda func: workflow(func, id=id)
    id = id or func.__name__
    return WorkflowInfo(id, func)


def env(value: dict=None, **kwargs):
    _update_field("env", value, **kwargs)

on = _OnUpdater()
