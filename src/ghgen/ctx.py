import contextlib
import dataclasses
import inspect
import itertools
import textwrap
import typing
from dataclasses import dataclass, fields, asdict
import pathlib

import inflection

from .expr import (
    Expr,
    on_error,
    contexts,
    FlatMap,
    Map,
    ErrorExpr,
    function,
    reftree,
    CallExpr,
)
from . import workflow, element
from .contexts import *


@dataclass
class Error:
    filename: str
    lineno: int
    workflow_id: str | None
    message: str

    def __str__(self):
        return f"{self.filename}:{self.lineno} [{self.workflow_id}] {self.message}"


_this_dir = pathlib.Path(__file__).parent


def _get_user_frame() -> typing.Any:
    frame = inspect.currentframe()

    for frame in iter(lambda: frame.f_back, None):
        filename = frame.f_code.co_filename
        # get first frame out of this app or contextlib (that wraps some of our functions)
        if not (
            pathlib.Path(filename).is_relative_to(_this_dir)
            or filename == contextlib.__file__
        ):
            break
    return frame


def _get_user_frame_info() -> inspect.Traceback:
    return inspect.getframeinfo(_get_user_frame())


@dataclass
class _Context(ContextBase):
    auto_job_reason: str | None = None
    errors: list[Error] = field(default_factory=list)

    def reset(self):
        self.reset_job()
        self.current_workflow = None
        self.current_workflow_id = None
        self.auto_job_reason = None
        self.errors = []

    def reset_job(self, job: Job | None = None, job_id: str | None = None):
        self.current_job = job
        self.current_job_id = job_id

    def empty(self) -> bool:
        return (
            not self.current_workflow
            and not self.current_job
            and not self.current_workflow_id
            and not self.current_job_id
            and not self.auto_job_reason
        )

    def make_error(self, message: str, id: str | None = None) -> Error:
        frame = _get_user_frame_info()
        return Error(
            frame.filename, frame.lineno, id or self.current_workflow_id, message
        )

    def error(self, message: str):
        error = self.make_error(message)
        if self.current_workflow:
            self.errors.append(error)
        else:
            # raise immediately
            raise GenerationError([error])

    @contextlib.contextmanager
    def build_workflow(self, id: str) -> typing.Generator[Workflow, None, None]:
        assert self.empty()
        self.current_workflow = Workflow()
        self.current_workflow_id = id
        with on_error(lambda message: self.error(message)):
            try:
                yield self.current_workflow
                if self.errors:
                    raise GenerationError(self.errors)
            finally:
                self.reset()

    @contextlib.contextmanager
    def build_job(self, id: str) -> typing.Generator[Job, None, None]:
        previous_job = self.current_job
        previous_job_id = self.current_job_id
        job = Job()
        self.reset_job(job, id)
        try:
            yield job
            if self.auto_job_reason:
                self.error(
                    f"explict job `{id}` cannot be created after already implicitly creating a job, which happened when setting `{self.auto_job_reason}`",
                )
            elif not self.current_workflow or previous_job:
                self.error(f"job `{id}` not created directly inside a workflow body")
            elif id in self.current_workflow.jobs:
                self.error(
                    f"job `{id}` already exists in workflow `{current_workflow_id()}`",
                )
            else:
                self.current_workflow.jobs[id] = job
        finally:
            self.reset_job(previous_job, previous_job_id)

    def auto_job(self, reason: str) -> Job:
        assert not self.current_job and not self.auto_job_reason
        wf = self.current_workflow
        job = Job(name=wf.name)
        if not wf.jobs:
            wf.jobs[self.current_workflow_id] = job
            self.reset_job(job, self.current_workflow_id)
            self.auto_job_reason = reason
        else:
            self.error(
                f"`{reason}` is a `job` field, but implicit job cannot be created because there are already jobs in the workflow",
            )
        return job


class _Updaters:
    @classmethod
    def instance(cls, reason: str) -> Workflow | Job: ...


@dataclass
class _Updater[**P, F]:
    field_init: typing.Callable[P, F]
    field: str | None = None
    owner: type[_Updaters] | typing.Self | None = None

    def __set_name__(self, owner: type[_Updaters], name: str):
        self.field = name
        self.owner = owner

    def _get_parent(self) -> typing.Any:
        match self.owner:
            case _Updater():
                return self.owner._apply()[2]
            case None:
                assert False
            case _:
                return self.owner.instance(self.field)

    def _apply(
        self, *args: P.args, **kwargs: P.kwargs
    ) -> tuple[typing.Any, F | None, F | None]:
        parent = self._get_parent()
        if parent is None:
            # error already reported, return a dummy value
            return None, None, None
        if (
            not kwargs
            and len(args) == 1
            and (isinstance(args[0], Expr) or args[0] == None)
        ):
            merged = value = args[0]
        else:
            current = getattr(parent, self.field)
            try:
                value = self.field_init(*args, **kwargs)
                merged = _merge(self.field, current, value)
            except (AssertionError, TypeError, ValueError):
                _ctx.error(f"illegal assignment to `{self.field}`")
                return parent, None, current
        setattr(parent, self.field, merged)
        return parent, value, merged

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> typing.Self:
        parent, value, _ = self._apply(*args, **kwargs)
        _ctx.validate(value, target=parent, field=self.field)
        return self.owner if isinstance(self.owner, _Updater) else self

    def __get__(self, instance: typing.Any, owner: type[_Updaters]) -> typing.Self:
        if instance is None:
            return self
        selfcls = typing.cast(_Updater, type(self))
        kwargs = dict(self.__dict__)
        kwargs["owner"] = instance
        return selfcls(**kwargs)


class _MapUpdater[**P, F](_Updater):
    value_init: typing.Callable[P, F]

    def __init__(self, value_init: typing.Callable[P, F], **kwargs):
        self.value_init = value_init
        kwargs.setdefault("field_init", dict)
        super().__init__(**kwargs)

    def __set_name__(self, owner, name):
        super().__set_name__(owner, f"{name}s")

    def __call__(self, key: str, *args: P.args, **kwargs: P.kwargs) -> typing.Self:
        return super().__call__({key: self.value_init(*args, **kwargs)})


class _WorkflowUpdaters(_Updaters):
    @classmethod
    def instance(cls, reason: str):
        if _ctx.auto_job_reason:
            _ctx.error(
                f"`{reason}` is a workflow field, and an implicit job was created when setting `{_ctx.auto_job_reason}`"
            )
            return None
        if _ctx.current_job:
            _ctx.error(
                f"`{reason}` is a workflow field, it cannot be set in job `{_ctx.current_job_id}`"
            )
            return None
        if not _ctx.current_workflow:
            _ctx.error(
                f"`{reason}` can only be set in a workflow. Did you forget a `@workflow` decoration?"
            )
            return None
        return _ctx.current_workflow

    class OnUpdater(_Updater):
        pull_request = _Updater(PullRequest)
        push = _Updater(Push)

        class WorkflowDispatchUpdater(_Updater):
            input = _MapUpdater(Input)

        workflow_dispatch = WorkflowDispatchUpdater(WorkflowDispatch)

        class WorkflowCallUpdater(WorkflowDispatchUpdater):
            secret = _MapUpdater(Secret)

        workflow_call = WorkflowCallUpdater(WorkflowCall)

    on = OnUpdater(On)


class _WorkflowOrJobUpdaters(_Updaters):
    @classmethod
    def instance(cls, reason: str):
        if not _ctx.current_workflow:
            _ctx.error(
                f"`{reason}` can only be set in a workflow or a job. Did you forget a `@workflow` decoration?"
            )
            return None
        return _ctx.current_job or _ctx.current_workflow

    name = _Updater(str)
    env = _Updater(dict)
    outputs = _Updater(dict)


class _JobUpdaters(_Updaters):
    @classmethod
    def instance(cls, reason: str):
        if not _ctx.current_workflow:
            _ctx.error(
                f"`{reason}` can only be set in a job or an implicit workflow job. Did you forget a `@workflow` decoration?"
            )
            return None
        if not _ctx.current_job:
            return _ctx.auto_job(reason)
        return _ctx.current_job

    runs_on = _Updater(str)

    class StrategyUpdater(_Updater):
        matrix = _Updater(Matrix)
        fail_fast = _Updater(lambda v=True: v)
        max_parallel = _Updater(int)

    strategy = StrategyUpdater(Strategy)
    steps = _Updater(list)

    class ContainerUpdater[**P, F](_Updater[P, F]):
        image = _Updater(str)
        credentials = _Updater(Credentials)
        env = _Updater(dict)
        ports = _Updater(list)
        volumes = _Updater(list)
        options = _Updater(list)

    container = ContainerUpdater(Container)

    service = _MapUpdater(Container)


_ctx = _Context()


def current() -> Workflow | Job | None:
    return _ctx.current_job or _ctx.current_workflow


def current_workflow_id() -> str | None:
    return _ctx.current_workflow_id


def _merge[T](field: str, lhs: T | None, rhs: T | None, recursed=False) -> T | None:
    try:
        match (lhs, rhs):
            case None, _:
                return rhs
            case _, None if not recursed:
                return None
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
                        f.name,
                        getattr(lhs, f.name),
                        getattr(rhs, f.name),
                        recursed=True,
                    )
                    for f in fields(lhs)
                }
                return type(lhs)(**data)
            case _:
                assert type(lhs) is type(rhs)
                return rhs
    except AssertionError as e:
        _ctx.error(f"cannot assign `{type(rhs).__name__}` to `{field}`")


class GenerationError(Exception):
    def __init__(self, errors: list[Error]):
        self.errors = errors

    def __str__(self):
        return "\n" + "\n".join(map(str, self.errors))


@dataclass
class WorkflowInfo:
    id: str
    spec: typing.Callable[..., None]
    errors: list[Error]
    file: pathlib.Path

    def instantiate(self) -> Workflow:
        with _ctx.build_workflow(self.id) as wf:
            for e in self.errors:
                e.workflow_id = e.workflow_id or current_workflow_id()
            _ctx.errors += self.errors
            self.spec()
            return wf


def workflow(
    func: typing.Callable[..., None] | None = None, *, id=None
) -> typing.Callable[[typing.Callable[..., None]], WorkflowInfo] | WorkflowInfo:
    if func is None:
        return lambda func: workflow(func, id=id)
    id = id or func.__name__
    errors = []
    return WorkflowInfo(id, func, errors, file=pathlib.Path(inspect.getfile(func)))


type JobCall = typing.Callable[..., None]


class _Job(ProxyExpr):
    def __call__(
        self, func: JobCall | None = None, *, id: str | None = None
    ) -> typing.Callable[[JobCall], Expr] | Expr:
        if func is None:
            return lambda func: self(func, id=id)
        id = id or func.__name__
        with _ctx.build_job(id) as j:
            func()
            return getattr(Contexts.needs, id)

    def _get_expr(self) -> Expr:
        return Contexts.job


job = _Job()
name = _WorkflowOrJobUpdaters.name
on = _WorkflowUpdaters.on
env = _WorkflowOrJobUpdaters.env
runs_on = _JobUpdaters.runs_on


def needs(*args: RefExpr) -> list[str]:
    prereqs = []
    unsupported = []
    for a in args:
        match a:
            case RefExpr(_segments=("needs", id)):
                prereqs.append(id)
            case _:
                unsupported.append(f"`{instantiate(a)}`")
    if unsupported:
        _ctx.error(
            f"`needs` only accepts job handles given by `@job`, got {", ".join(unsupported)}"
        )
    else:
        # this checks and autofills needs
        _ctx.validate([*args])
    return prereqs


class _StrategyUpdater(ProxyExpr):
    def __call__(self, *args, **kwargs):
        return _JobUpdaters.strategy(*args, **kwargs)

    def _get_expr(self) -> Expr:
        return Contexts.strategy

    class FailFastUpdater(ProxyExpr):
        def __call__(self, value: Value[bool] = True):
            return _JobUpdaters.strategy.fail_fast(value)

        def _get_expr(self) -> Expr:
            return Contexts.strategy.fail_fast

    fail_fast = FailFastUpdater()

    class MaxParallelUpdater(ProxyExpr):
        def __call__(self, value: Value[int] = True):
            return _JobUpdaters.strategy.max_parallel(value)

        def _get_expr(self) -> Expr:
            return Contexts.strategy.max_parallel

    max_parallel = MaxParallelUpdater()

    def matrix(self, *args, **kwargs):
        return _JobUpdaters.strategy.matrix(*args, **kwargs)

    job_index: RefExpr
    job_total: RefExpr


strategy = _StrategyUpdater()
container = _JobUpdaters.container
service = _JobUpdaters.service


def _allocate_id(
    prefix: str, is_free: typing.Callable[[str], bool], *, start_from_one: bool = False
) -> str:
    prefix = Element._key(prefix)  # replace underscores with dashes
    if not start_from_one and is_free(prefix):
        return prefix
    return next(
        (id for id in (f"{prefix}-{i}" for i in itertools.count(1)) if is_free(id))
    )


def _get_var_name(pred: typing.Callable[[object], bool]) -> str | None:
    frame = _get_user_frame()
    return next((var for var, value in frame.f_locals.items() if pred(value)), None)


@dataclass
class _InputUpdater(ProxyExpr):
    _input: Input | None = None

    def __init__(self, input: Input | None = None):
        super().__init__()
        self._input = input

    def __call__(self, description: str | None = None) -> typing.Self:
        return self.description(description)

    @property
    def _triggers(self) -> tuple[WorkflowCall | None, WorkflowDispatch | None]:
        if not _ctx.current_workflow:
            return None, None
        return (
            _ctx.current_workflow.on.workflow_call,
            _ctx.current_workflow.on.workflow_dispatch,
        )

    def _ensure_input(self) -> typing.Self:
        if self._input:
            return self
        ret = _InputUpdater(Input())
        if not _ctx.current_workflow or _ctx.current_job:
            _ctx.error("`input` can only be used in a workflow")
            return ret
        if all(t is None for t in self._triggers):
            _ctx.error(
                "`input` must be used after setting either `on.workflow_call` or `on.workflow_dispatch`"
            )
            return ret
        for t in self._triggers:
            if t is not None:
                if t.inputs is None:
                    t.inputs = []
                t.inputs.append(ret._input)
        return ret

    def _finalize(self):
        assert self._input
        try:
            self._input.__post_init__()
        except ValueError as e:
            _ctx.error(e.args[0])

    def id(self, id: str) -> typing.Self:
        if self._input is None:
            self = self()
        if self._input.id is not None:
            _ctx.error(f"id was already specified for this input as `{self._input.id}`")
        elif any(i.id == id for t in self._triggers if t is not None for i in t.inputs):
            _ctx.error(f"id `{id}` was already specified for an input")
        else:
            self._input.id = id
        return self

    def required(self, value: bool = True) -> typing.Self:
        ret = self._ensure_input()
        ret._input.required = value
        return ret

    def default(self, value: typing.Any):
        ret = self._ensure_input()
        ret._input.default = value
        ret._finalize()
        return ret

    def description(self, value: str | None) -> typing.Self:
        ret = self._ensure_input()
        assert ret._input is not None
        ret._input.description = value
        return ret

    def type(self, value: typing.Any) -> typing.Self:
        ret = self._ensure_input()
        ret._input.type = value
        ret._finalize()
        return ret

    def options(
        self, seq_or_first: str | typing.Iterable[str] | None, *rest: str
    ) -> typing.Self:
        match seq_or_first, rest:
            case str(), _:
                seq = [seq_or_first] + list(rest)
            case typing.Iterable(), ():
                seq = list(seq_or_first)
            case None, ():
                seq = seq_or_first
            case _:
                _ctx.error(
                    "`options` must be given either a sequence of strings or string arguments"
                )
                return self
        ret = self._ensure_input()
        ret._input.options = seq
        ret._finalize()
        return ret

    def ensure_id(self) -> str:
        ret = self._ensure_input()
        if ret._input.id is None:
            id = _get_var_name(lambda v: v is ret)
            ret._input.id = _allocate_id(
                id or "input",
                lambda id: all(i.id != id for i in _ctx.current_workflow.inputs),
                start_from_one=id is None,
            )
        return ret._input.id

    def _get_expr(self) -> Expr:
        if self._input is None:
            return ~ErrorExpr("`input` alone cannot be used in an expression")
        id = self.ensure_id()
        return getattr(Contexts.inputs, id)


def _ensure_step_id(s: Step) -> str:
    if s.id is None:
        id = _get_var_name(lambda v: isinstance(v, _StepUpdater) and v._step is s)
        s.id = _allocate_id(
            id or "step",
            lambda id: all(s.id != id for s in _ctx.current_job.steps),
            start_from_one=id is None,
        )
    return s.id


@dataclass
class _StepUpdater(ProxyExpr):
    _step: Step | None = None

    def __init__(self, step: Step | None = None):
        super().__init__()
        self._step = step

    def __call__(self, name: Value[str]) -> typing.Self:
        return self.name(name)

    def _get_expr(self) -> Expr:
        if self._step is None:
            return ~ErrorExpr("`step` alone cannot be used in an expression")
        id = self.ensure_id()
        return getattr(Contexts.steps, id)

    def _ensure_step(self) -> typing.Self:
        if self._step is not None:
            return self
        _JobUpdaters.steps([Step()])
        step = (
            _ctx.current_job and _ctx.current_job.steps and _ctx.current_job.steps[-1]
        )
        return _StepUpdater(step)

    def _ensure_run_step(self) -> typing.Self:
        ret = self._ensure_step()
        if ret._step.uses or ret._step.with_:
            _ctx.error("cannot turn a `use` step into a `run` one")
        return ret

    def _ensure_use_step(self) -> typing.Self:
        ret = self._ensure_step()
        if ret._step.run:
            _ctx.error("cannot turn a `run` step into a `use` one")
        return ret

    def id(self, id: str) -> typing.Self:
        ret = self._ensure_step()
        if ret._step.id:
            _ctx.error(f"id was already specified for this step as `{ret._step.id}`")
        elif any(s.id == id for s in _ctx.current_job.steps):
            _ctx.error(f"id `{id}` was already specified for a step")
        else:
            ret._step.id = id
        return ret

    def name(self, name: Value[str]) -> typing.Self:
        ret = self._ensure_step()
        _ctx.validate(name, target=ret._step, field="name")
        ret._step.name = name
        return ret

    def if_(self, condition: Value[bool]) -> typing.Self:
        ret = self._ensure_step()
        _ctx.validate(condition, target=ret._step, field="if_")
        ret._step.if_ = condition
        return ret

    def env(self, *args, **kwargs) -> typing.Self:
        ret = self._ensure_run_step()
        value = dict(*args, **kwargs)
        _ctx.validate(value, target=ret._step, field="env")
        ret._step.env = (ret._step.env or {}) | value
        return ret

    def run(self, code: Value[str]):
        ret = self._ensure_run_step()
        _ctx.validate(code, target=ret._step, field="run")
        if isinstance(code, str):
            code = textwrap.dedent(code.strip("\n"))
        ret._step.run = code
        return ret

    def uses(self, source: Value[str], **kwargs):
        ret = self._ensure_use_step()
        _ctx.validate(source, target=ret._step, field="uses")
        ret._step.uses = source
        if isinstance(source, str) and not ret._step.name:
            try:
                _, _, action_name = source.rpartition("/")
                action_name, _, _ = action_name.partition("@")
                action_name = inflection.humanize(inflection.titleize(action_name))
                ret._step.name = action_name
            except Exception:
                pass
        if kwargs:
            ret.with_(kwargs)
        return ret

    def continue_on_error(self, value: Value[bool] = True) -> typing.Self:
        ret = self._ensure_step()
        _ctx.validate(value, target=ret._step, field="continue_on_error")
        ret._step.continue_on_error = value
        return ret

    def with_(self, *args, **kwargs) -> typing.Self:
        ret = self._ensure_use_step()
        value = dict(*args, **kwargs)
        _ctx.validate(value, target=ret._step, field="with_")
        ret._step.with_ = (ret._step.with_ or {}) | value
        return ret

    def returns(self, *args: str, **kwargs: Value[str]) -> typing.Self:
        ret = self._ensure_run_step()
        outs = list(args)
        outs.extend(a for a in kwargs if a not in args)
        _ctx.validate(kwargs, target=ret._step, field="outputs")
        ret._step.outputs = ret._step.outputs or []
        ret._step.outputs += outs
        # TODO: support other shells than bash
        if kwargs:
            # TODO: handle quoting?
            out_code = "\n".join(
                f"echo {k}={v} >> $GITHUB_OUTPUTS" for k, v in kwargs.items()
            )
            ret._step.run = (
                f"{ret._step.run}\n{out_code}" if ret._step.run else out_code
            )
        return ret

    def needs(self, *jobs: RefExpr) -> typing.Self:
        jobs = needs(*jobs)
        ret = self._ensure_step()
        ret._step.needs = jobs
        return ret

    def ensure_id(self) -> str:
        return _ensure_step_id(self._ensure_step()._step)


input = _InputUpdater()
step = _StepUpdater()
run = step.run
use = step.uses


def _dump_step_outputs(s: Step):
    id = _ensure_step_id(s)
    if s.outputs:
        outputs = getattr(steps, id).outputs
        _WorkflowOrJobUpdaters.outputs((o, getattr(outputs, o)) for o in s.outputs)
    else:
        _ctx.error(
            f"step `{id}` passed to `outputs`, but no outputs were declared on it. Use `returns()` to do so",
        )


def _dump_job_outputs(id: str, j: Job):
    if j.outputs:
        outputs = getattr(Contexts.jobs, id).outputs
        _WorkflowOrJobUpdaters.outputs((o, getattr(outputs, o)) for o in j.outputs)
    else:
        _ctx.error(
            f"job `{id}` passed to `outputs`, but no outputs were declared on it. Use `outputs()` to do so",
        )


def outputs(*args: typing.Literal["*"] | RefExpr | _StepUpdater, **kwargs: typing.Any):
    instance = _WorkflowOrJobUpdaters.instance("outputs")
    unsupported = lambda: _ctx.error(
        f'unsupported unnamed output `{instantiate(arg)}`, must be `"*"`, a context field or a step'
    )
    for arg in args:
        match instance, arg:
            case Workflow(), RefExpr(_segments=("needs", id)) if id in instance.jobs:
                _dump_job_outputs(id, instance.jobs[id])
            case _, RefExpr():
                key = arg._segments[-1]
                _WorkflowOrJobUpdaters.outputs(((key, arg),))
            case Job(), _StepUpdater():
                _dump_step_outputs(arg._step)
            case Workflow(), _StepUpdater():
                _ctx.error(f"a step cannot be returned as a workflow output")
            case _, Expr():  # this must be done to avoid comparing Expr with "*"
                unsupported()
            case Job(), "*":
                for s in instance.steps:
                    if s.outputs:
                        _dump_step_outputs(s)
            case Workflow(), "*":
                for id, j in instance.jobs.items():
                    if j.outputs:
                        _dump_job_outputs(id, j)
            case _:
                unsupported()
    _WorkflowOrJobUpdaters.outputs(kwargs)


always = function("always", 0)
cancelled = function("cancelled", 0)
fromJson = function("fromJson")
contains = function("contains", 2)
