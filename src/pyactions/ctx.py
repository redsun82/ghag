import contextlib
import inspect
import typing
import warnings
from dataclasses import dataclass, fields, asdict, field
import dataclasses
import threading

from .element import Element
from .expr import Value
from .workflow import *


@dataclass
class Error:
    filename: str
    lineno: int
    workflow_id: str
    message: str

    def __str__(self):
        return f"{self.filename}:{self.lineno} [{self.workflow_id}] {self.message}"


@dataclass
class _Context(threading.local):
    current: Workflow | Job | None = None
    current_workflow_id: str | None = None
    auto_job_reason: str | None = None
    errors: list[Error] = field(default_factory=list)

    def reset(self):
        self.current = None
        self.current_workflow_id = None
        self.auto_job_reason = None
        self.errors = []

    def empty(self) -> bool:
        return not self.current and not self.current_workflow_id and not self.errors

    def error(self, message: str, level: int = 2):
        frame = inspect.currentframe()
        for _ in range(level):
            frame = frame.f_back
        frame = inspect.getframeinfo(frame)
        error = Error(frame.filename, frame.lineno, self.current_workflow_id, message)
        self.errors.append(error)

    def check(self, cond: bool, message: str, level=2):
        if not cond:
            self.error(message, level)
        return cond


_ctx = _Context()


@contextlib.contextmanager
def _build_workflow(id: str):
    assert _ctx.empty()
    _ctx.current = Workflow()
    _ctx.current_workflow_id = id
    try:
        yield _ctx.current
        if _ctx.errors:
            raise GenerationError(_ctx.errors)
    finally:
        _ctx.reset()


@contextlib.contextmanager
def _build_job(id: str):
    parent = current()
    job = _ctx.current = Job()
    try:
        yield _ctx.current
        if not isinstance(parent, Workflow):
            if _ctx.auto_job_reason:
                _ctx.error(
                    f"explict job `{id}` cannot be created after already implicitly creating a job, which happened when setting `{_ctx.auto_job_reason}`",
                    level=4,
                )
            else:
                _ctx.error(
                    f"job `{id}` not created directly inside a workflow body", level=4
                )
            if current_workflow_id() is None:
                # we aren't even in a workflow, raise immediately
                errors = _ctx.errors
                _ctx.errors = []
                raise GenerationError(errors)
        elif id in parent.jobs:
            _ctx.error(
                f"job `{id}` already exists in workflow `{current_workflow_id()}`",
                level=4,
            )
        else:
            parent.jobs[id] = job
    finally:
        _ctx.current = parent


def _start_auto_job_reason(field: str) -> Job:
    assert isinstance(current(), Workflow) and not _ctx.auto_job_reason
    job = Job(name=current().name)
    if not current().jobs:
        current().jobs[current_workflow_id()] = job
        _ctx.current = job
        _ctx.auto_job_reason = field
    else:
        _ctx.error(
            f"`{field}` is a `job` field, but implicit job cannot be created because there are already jobs in the workflow",
            level=5,
        )
    return job


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
    instance = instance or current()
    ret = _try_get_field(field, instance)
    if ret is None and isinstance(instance, Workflow):
        ret = _try_get_field(field, Job)
        assert ret, f"field {field} not found in current instance (Workflow or Job)"
        return _start_auto_job_reason(field), ret
    if not ret:
        _ctx.error(
            f"`{field}` is not a {type(instance).__name__.lower()} field", level=4
        )
        if isinstance(instance, Job) and _ctx.auto_job_reason:
            _ctx.errors[
                -1
            ].message += f", and an implicit job was created when setting `{_ctx.auto_job_reason}`"
    return instance, ret


def _merge[T](field: str, level: int, lhs: T | None, rhs: T | None) -> T | None:
    try:
        match (lhs, rhs):
            case None, _:
                return rhs
            case _, None:
                return lhs
            case dict(), dict():
                return lhs | rhs
            case list(), list():
                return lhs + rhs
            case list(), str() | bytes():
                assert False
            case list(), typing.Iterable():
                return lhs + list(rhs)
            case Element(), Element():
                assert type(lhs) is type(rhs)
                data = {
                    f.name: _merge(
                        f.name, level + 1, getattr(lhs, f.name), getattr(rhs, f.name)
                    )
                    for f in fields(lhs)
                }
                return type(lhs)(**data)
            case _:
                assert type(lhs) is type(rhs)
                return rhs
    except AssertionError as e:
        _ctx.error(
            f"cannot assign `{type(rhs).__name__}` to `{field}`", level=level + 2
        )


def _update_field(field: str, *args, **kwargs) -> typing.Any:
    instance, f = _get_field(field)
    if not f:
        return None
    current_value = getattr(instance, field) or f.type()
    value = args[0] if len(args) == 1 and not kwargs else f.type(*args, **kwargs)
    value = _merge(field, 2, current_value, value)
    setattr(instance, field, value)
    return value


def _update_subfield(field: str, subfield: str, *args, **kwargs) -> typing.Any:
    instance, f = _get_field(field)
    if not f:
        return None
    value = f.type()
    _, sf = _get_field(subfield, value)
    if not sf:
        return None
    subvalue = args[0] if len(args) == 1 and not kwargs else sf.type(*args, **kwargs)
    setattr(value, subfield, subvalue)
    value = _merge(subfield, 2, getattr(instance, field), value)
    setattr(instance, field, value)
    return value


class GenerationError(Exception):
    def __init__(self, errors: list[Error]):
        self.errors = errors

    def __str__(self):
        return "\n".join(map(str, self.errors))


@dataclass
class WorkflowInfo:
    id: str
    spec: typing.Callable[[], None]

    def instantiate(self) -> Workflow:
        with _build_workflow(self.id) as wf:
            wf.name = self.spec.__doc__
            self.spec()
            return wf


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
    with _build_job(id) as job:
        job.name = func.__doc__
        func()


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
