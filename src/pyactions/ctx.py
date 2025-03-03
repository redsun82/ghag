import typing
import warnings
from dataclasses import dataclass, fields, asdict
import dataclasses
import threading

from .element import Element
from .expr import Value
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


def _try_get_field(field: str, instance: Element | None = None) -> dataclasses.Field:
    instance = instance or current()
    return next((f for f in fields(instance or current()) if f.name == field), None)


def _get_field(
    field: str, instance: Element | None = None
) -> tuple[Element, dataclasses.Field]:
    ret = _try_get_field(field, instance)
    if ret is None and instance is None and isinstance(current(), Workflow):
        instance = current().jobs.setdefault(
            current_workflow_id(), Job(name=current().name)
        )
        ret = _try_get_field(field, instance)
        assert ret, f"field {field} not found in current instance (Workflow or Job)"
        return instance, ret
    assert (
        ret
    ), f"field {field} not found in current instance ({type(instance).__name__})"
    return instance or current(), ret


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


def _update_field(field: str, *args, **kwargs) -> typing.Any:
    instance, f = _get_field(field)
    current_value = getattr(instance, field)
    value = args[0] if len(args) == 1 and not kwargs else f.type(*args, **kwargs)
    value = _merge(current_value, value)
    setattr(instance, field, value)
    return value


def _update_subfield(field: str, subfield: str, *args, **kwargs) -> typing.Any:
    instance, f = _get_field(field)
    value = f.type()
    _, sf = _get_field(subfield, value)
    subvalue = args[0] if len(args) == 1 and not kwargs else sf.type(*args, **kwargs)
    setattr(value, subfield, subvalue)
    value = _merge(getattr(instance, field), value)
    setattr(instance, field, value)
    return value


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


def runs_on(value: Value):
    _update_field("runs_on", value)


class _OnUpdater:
    def pull_request(self, **kwargs) -> typing.Self:
        _update_subfield("on", "pull_request", **kwargs)
        return self

    def workflow_dispatch(self, **kwargs) -> typing.Self:
        _update_subfield("on", "workflow_dispatch", **kwargs)
        return self


on = _OnUpdater()


class _StrategyUpdater:
    def matrix(self, **kwargs) -> typing.Self:
        _update_subfield("strategy", "matrix", **kwargs)
        return self

    def fail_fast(self, value: Value[bool] = True) -> typing.Self:
        _update_subfield("strategy", "fail_fast", value)
        return self

    def max_parallel(self, value: Value[int]) -> typing.Self:
        _update_subfield("strategy", "max_parallel", value)
        return self


strategy = _StrategyUpdater()


@dataclass
class _StepUpdater:
    steps: list[Step] | None = None

    def __call__(self, name: Value[str]) -> typing.Self:
        return self.name(name)

    @property
    def step(self) -> Step | None:
        return self.steps[-1] if self.steps else None

    def _ensure_step(self) -> typing.Self:
        if self.step is not None:
            return self
        steps = _update_field("steps", [Step()])
        return _StepUpdater(steps)

    def _ensure_run_step(self) -> typing.Self:
        ret = self._ensure_step()
        match ret.step:
            case RunStep():
                pass
            case UseStep():
                assert False, "cannot turn a `use` step into a `run` one"
            case _:
                ret.steps[-1] = RunStep(**asdict(ret.step))
        return ret

    def _ensure_use_step(self) -> typing.Self:
        ret = self._ensure_step()
        match ret.step:
            case RunStep():
                assert False, "cannot turn a `run` step into a `use` one"
            case UseStep():
                pass
            case _:
                ret.steps[-1] = UseStep(**asdict(ret.step))
        return ret

    def name(self, name: Value[str]) -> typing.Self:
        ret = self._ensure_step()
        ret.step.name = name
        return ret

    def if_(self, condition: Value[bool]) -> typing.Self:
        ret = self._ensure_step()
        ret.step.if_ = condition
        return ret

    def env(self, *args, **kwargs) -> typing.Self:
        ret = self._ensure_run_step()
        ret.step.env = (ret.step.env or {}) | dict(*args, **kwargs)
        return ret

    def run(self, code: Value[str]):
        ret = self._ensure_run_step()
        ret.step.run = code
        return ret

    def use(self, source: Value[str], **kwargs):
        ret = self._ensure_use_step()
        ret.step.use = source
        if kwargs:
            ret.with_(kwargs)
        return ret

    def with_(self, *args, **kwargs) -> typing.Self:
        ret = self._ensure_use_step()
        ret.step.with_ = (ret.step.with_ or {}) | dict(*args, **kwargs)
        return ret

step = _StepUpdater()
run = step.run
use = step.use