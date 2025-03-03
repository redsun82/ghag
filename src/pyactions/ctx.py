import typing
import warnings
from dataclasses import dataclass, fields, asdict
import dataclasses
import threading

from .element import Element
from .workflow import *


@dataclass
class _Context(threading.local):
    current: Workflow | Job | None = None
    current_workflow_id: str | None = None


_ctx = _Context()


def current() -> Workflow | Job | None:
    return _ctx.current


def current_workflow_id() -> str | None:
    return _ctx.current_workflow_id


def _get_field(field: str, instance: Element | None = None) -> dataclasses.Field:
    instance = instance or current()
    ret = next((f for f in fields(instance or current()) if f.name == field), None)
    assert (
        ret
    ), f"field {field} not found in current instance ({type(instance).__name__})"
    return ret


def _merge[T](lhs: T | None, rhs: T | None) -> T | None:
    match (lhs, rhs):
        case None, _:
            return rhs
        case _, None:
            return lhs
        case dict(), dict():
            return lhs | rhs
        case list(), list():
            return lhs + rhs
        case Element(), Element():
            assert type(lhs) is type(rhs)
            data = {
                f.name: _merge(getattr(lhs, f.name), getattr(rhs, f.name))
                for f in fields(lhs)
            }
            return type(lhs)(**data)
        case _:
            assert type(lhs) is type(rhs)
            return rhs


def _update_field(field: str, *args, **kwargs):
    f = _get_field(field)
    current_value = getattr(current(), field)
    value = args[0] if len(args) == 1 and not kwargs else f.type(*args, **kwargs)
    setattr(current(), field, _merge(current_value, value))


def _update_subfield(field: str, subfield: str, *args, **kwargs):
    f = _get_field(field)
    value = f.type()
    sf = _get_field(subfield, value)
    subvalue = args[0] if len(args) == 1 and not kwargs else sf.type(*args, **kwargs)
    setattr(value, subfield, subvalue)
    value = _merge(getattr(current(), field), value)
    setattr(current(), field, value)


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
        ret = _ctx.current = Workflow(name=self.spec.__doc__)
        _ctx.current_workflow_id = self.id
        self.spec()
        _ctx.current = None
        _ctx.current_workflow_id = None
        return ret


def workflow(
    func: typing.Callable[[], None] | None = None, *, id=None
) -> typing.Callable[[typing.Callable[[], None]], WorkflowInfo] | WorkflowInfo:
    if func is None:
        return lambda func: workflow(func, id=id)
    id = id or func.__name__
    return WorkflowInfo(id, func)


def job(
    func: typing.Callable[[], None] | None = None, *, id=None
) -> typing.Callable[[typing.Callable[[], None]], None] | None:
    if func is None:
        return lambda func: job(func, id=id)
    id = id or func.__name__
    wf = current()
    assert isinstance(wf, Workflow)
    if id in wf.jobs:
        warnings.warn(f"Overwriting job {id!r} in workflow {current_workflow_id()!r}")
    job = _ctx.current = wf.jobs[id] = Job(name=func.__doc__)
    func()
    _ctx.current = wf


def name(value: str):
    _update_field("name", value)


def env(*args, **kwargs):
    _update_field("env", *args, **kwargs)


on = _OnUpdater()
