import collections.abc
import contextlib
import dataclasses
import enum
import inspect
import itertools
import textwrap
import types
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

    def process_final_workflow(self):
        w = self.current_workflow
        assert w is not None
        id = self.current_workflow_id
        for j_id, j in w.jobs.items():
            for i, s in enumerate(j.steps or (), 1):
                if s.with_ and not s.uses is not None:
                    _ctx.error(
                        f"step `{s.id or i}` in job `{j_id}` has a `with` but no `uses`"
                    )
                elif s.uses is None and s.run is None:
                    s.run = ""
            if j.steps and j.runs_on is None:
                j.runs_on = default_runner
        self.check(
            w.on.has_triggers,
            f"workflow `{id}` must have at least one trigger",
        ) and self.check(
            w.jobs,
            f"workflow `{id}` must have at least one job",
        )
        if w.on.workflow_call and w.on.workflow_call.outputs:
            unset_outputs = [
                o.id
                for o in w.on.workflow_call.outputs
                if o.id is not None and o.value is None
            ]
            self.check(
                not unset_outputs,
                f"workflow `{id}` has no value set for {', '.join(unset_outputs)}, use `outputs` to set them",
            )

    @contextlib.contextmanager
    def build_workflow(self, id: str) -> typing.Generator[Workflow, None, None]:
        assert self.empty()
        self.current_workflow = Workflow()
        self.current_workflow_id = id
        with on_error(lambda message: self.error(message)):
            try:
                yield self.current_workflow
                self.process_final_workflow()
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
            yield job
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


type _Path = tuple[str | int, ...]


def _type(path: _Path) -> type | None:
    t = Workflow
    for p in path:
        o = typing.get_origin(t)
        match p:
            case str() if issubclass(o or t, Element):
                f = next((f for f in fields(o or t) if f.name == p), None)
                assert f is not None, f"no `{p}` field in `{t.__name__}`"
                t = f.type
                if typing.get_origin(t) is types.UnionType:
                    t = typing.get_args(t)[0]
            case str() if o is dict:
                t = typing.get_args(t)[1]
            case int() if o is list and p >= 0:
                t = typing.get_args(t)[0]
            case _:
                assert False, f"unexpected access by `{p!r}` in `{t}`"
        if t is Value:
            t = str
    return t

<<<<<<< Updated upstream

@dataclass
class _Appender:
    index: int | None = None


=======
>>>>>>> Stashed changes
def _ensure_element_with_type(
    start: Workflow | Job, start_type: type, path: tuple[str | int, ...]
) -> tuple[typing.Any, type]:
    assert start is not None
    e = start
    t = start_type
    for p in path:
        match e, p:
            case _, str() if issubclass(t, Element):
                f = next((f for f in fields(t) if f.name == p), None)
                assert f is not None, f"no `{p}` field in `{t.__name__}`"
                t = f.type
                if typing.get_origin(t) is types.UnionType:
                    t = typing.get_args(t)[0]
                next_e = getattr(e, p)
                if next_e is None:
                    setattr(e, p, t())
                    e = getattr(e, p)
                else:
                    e = next_e
            case (list(), int() as i) | (list(), _Appender(index=int() as i)) if (
                0 <= i <= len(e)
            ):
                t = typing.get_args(t)[0]
                if i == len(e):
                    e.append(t())
                e = e[i]
            case list(), _Appender(index=None):
                t = typing.get_args(t)[0]
                p.index = len(e)
                e.append(t())
                e = e[-1]
            case dict(), str():
                t = typing.get_args(t)[1]
                e = e.setdefault(p, t())
            case _:
                assert False, f"unexpected access by `{p!r}` in `{t}`"
    return e, t


def _ensure_element(start: Workflow | Job, path: tuple[str | int, ...]) -> typing.Any:
    return _ensure_element_with_type(start, type(start), path)[0]


def _get_workflow(path: str) -> Workflow:
    if not isinstance(path, str):
        raise ValueError("PROUT")
    if _ctx.current_workflow is None or _ctx.current_job is not None:
        _ctx.error(f"`{path}` must be used in a workflow")
        return Workflow()
    return _ctx.current_workflow


def _get_job(path: str) -> Job:
    if _ctx.current_job is None:
        if _ctx.current_workflow:
            return _ctx.auto_job(path)
        _ctx.error(f"`{path}` must be used in a job")
        return Job()
    return _ctx.current_job


def _get_job_or_workflow(path: str) -> Workflow | Job:
    ret = _ctx.current_job or _ctx.current_workflow
    if ret is None:
        _ctx.error(f"`{path}` must be used in a workflow or a job")
        return Workflow()
    return ret


def _update_element(
    value: typing.Any,
    start: typing.Any,
    *path: str | int,
):
    prefix, field = path[:-1], path[-1]
    parent = _ensure_element(start, prefix)
    _ctx.validate(value, target=parent, field=field)
    old_value = getattr(parent, field)
    new_value = _merge(".".join(map(str, path)), old_value, value)
    if new_value is not None:
        setattr(parent, field, new_value)


@dataclasses.dataclass
class _NewUpdater[T]:
    _start: typing.Callable[[str], typing.Any]
    _path: tuple[str | int, ...]
    _cached_element: T | None = None
    _cached_parent: typing.Any = None
    _cached_parent_type: typing.Any = None

    @property
    def _log_path(self) -> str:
        return ".".join(map(str, self._path))

    @property
    def _element(self) -> T:
        if self._cached_element is None:
            parent = self._parent
            self._cached_element, _ = _ensure_element_with_type(
                parent, self._cached_parent_type, (self._path[-1],)
            )
        return self._cached_element

    def _get_parent_with_type(self) -> tuple[typing.Any, type]:
        start = self._start(self._log_path)
        return _ensure_element_with_type(start, type(start), self._path[:-1])

    @property
    def _parent(self) -> typing.Any:
        if self._cached_parent is None:
            self._cached_parent, self._cached_parent_type = self._get_parent_with_type()
        return self._cached_parent

    def _update[U](
        self,
        field: str | int,
        ty: typing.Callable[[str, U], typing.Any],
        value: U,
    ) -> typing.Self:
        ret = self._ensure()
        value = ty(field, value)
        el = ret._element
        _ctx.validate(value, target=el, field=field)
        old_value = getattr(el, field)
        new_value = _merge(field, old_value, value)
        if new_value is not None:
            setattr(el, field, new_value)
        return ret

    def _update_list[U](
            self,
            field: str | int,
            ty: typing.Callable[[str, U], typing.Any],
            value: U,
    ):
        self._sub_updater(_NewUpdater[list[U]], field)._update(_Appender(), ty, value)

    def _ensure(self) -> typing.Self:
        start = self._start(self._log_path)
        _ensure_element(start, self._path)
        return self

    def _sub_updater[U](self, ty: type[U], *fields: str) -> U:
        return ty(self._start, self._path + fields)


@dataclasses.dataclass
class _IdElementUpdater[T](_NewUpdater[T]):
    @property
    def _log_path(self) -> str:
        return ".".join(map(str, self._path[:-1]))[:-1]

    @property
    def _instantiated(self) -> bool:
        return self._cached_element is not None

    def _ensure(self) -> typing.Self:
        if self._instantiated:
            return self
        parent_list, parent_type = self._get_parent_with_type()
        assert isinstance(parent_list, list)
        ret = type(self)(
            self._start,
            self._path[:-1] + (len(parent_list),),
            _cached_parent=parent_list,
            _cached_parent_type=parent_type,
        )
        _ = ret._element
        return ret

    def id(self, id: str | None) -> typing.Self:
        ret = self._ensure()
        el = ret._element
        if id is None:
            ret._update("id", _value, None)
        elif el.id is not None:
            _ctx.error(f"id was already specified for this element as `{el.id}`")
        elif any(s.id == id for s in ret._parent):
            parent_path = ".".join(map(str, ret._path[:-1]))
            _ctx.error(f"id `{id}` already used in `{parent_path}`")
        else:
            ret._update("id", _value, id)
        return ret

    def ensure_id(self) -> str:
        ret = self._ensure()
        el = ret._element
        if el.id is None:
            id = _get_var_name(lambda v: v is ret)
            parent = ret._parent
            el.id = _allocate_id(
                # remove `s` from name of parent list to get default name
                id or ret._path[-2][:-1],
                lambda id: all(i.id != id for i in parent),
                start_from_one=id is None,
            )
        return el.id


def _seq(
    field: str, args: tuple[Value | typing.Iterable[str] | None, ...]
) -> list[Value] | None:
    if not args or (len(args) == 1 and args[0] is None):
        return None
    ret = []
    for arg in args:
        match arg:
            case str() | bool() | int() | float() | Expr():
                ret.append(arg)
            case collections.abc.Iterable():
                ret.extend(arg)
            case _:
                _ctx.error(f"`{field}` cannot accept element `{arg}`")
    return ret

type ValueMapping = dict[str, Value] | typing.Iterable[tuple[str, Value]] | None

def _map(
    field: str,
    args: tuple[
        ValueMapping, dict[str, Value]
    ],
) -> dict[str, Value] | None:
    arg, kwargs = args
    if arg is None and not kwargs:
        return None
    elif arg is None:
        return kwargs
    try:
        return dict(arg, **kwargs)
    except (TypeError, ValueError) as e:
        _ctx.error(f"illegal assignment to `{field}`: {e}")
        return None


def _field_map(
    field: str,
    args: tuple[
        ValueMapping, dict[str, Value]
    ],
) -> dict[str, Value] | None:
    args, kwargs = args
    kwargs = {Element._key(k): v for k, v in kwargs.items() if v is not None}
    ret = _map(field, (args, kwargs))
    return ret


def _value[T](field: str, value: T) -> T:
    return value


def _text[T](field: str, value: T) -> T:
    if isinstance(value, str):
        value = textwrap.dedent(value.strip("\n"))
    return value


class _Updaters:
    @classmethod
    def instance(cls, reason: str) -> Workflow | Job: ...


@dataclass
class _Updater[**P, F]:
    field_init: typing.Callable[P, F]
    field: str | None = None
    owner: type[_Updaters] | typing.Self | None = None

    def __set_name__(self, owner: type[_Updaters], name: str):
        self.field = name.lstrip("_")
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
            except (AssertionError, TypeError, ValueError) as e:
                _ctx.error(f"illegal assignment to `{self.field}` ({e})")
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


class _OnUpdater(_NewUpdater):
    class _PullRequestOrPush(_NewUpdater):
        def branches(self, *branches: str | typing.Iterable[str]) -> typing.Self:
            return self._update("branches", _seq, branches)

        def ignore_branches(self, *branches: str | typing.Iterable[str]) -> typing.Self:
            return self._update("ignore_branches", _seq, branches)

        def paths(self, *paths: str | typing.Iterable[str]) -> typing.Self:
            return self._update("paths", _seq, paths)

        def ignore_paths(self, *paths: str | typing.Iterable[str]) -> typing.Self:
            return self._update("ignore_paths", _seq, paths)

    class _PullRequest(_PullRequestOrPush):
        def types(self, *types: str | typing.Iterable[str]) -> typing.Self:
            return self._update("types", _seq, types)

        def __call__(
            self,
            *,
            branches: typing.Iterable[str] | None = None,
            ignore_branches: typing.Iterable[str] | None = None,
            paths: typing.Iterable[str] | None = None,
            ignore_paths: typing.Iterable[str] | None = None,
            types: typing.Iterable[str] | None = None,
        ) -> "_OnUpdater":
            self.branches(branches).ignore_branches(ignore_branches).paths(
                paths
            ).ignore_paths(ignore_paths).types(types)
            return on

    class _Push(_PullRequestOrPush):
        def tags(self, *tags: str | typing.Iterable[str]) -> typing.Self:
            return self._update("tags", _seq, tags)

        def ignore_tags(self, *tags: str | typing.Iterable[str]) -> typing.Self:
            return self._update("ignore_tags", _seq, tags)

        def __call__(
            self,
            *,
            branches: typing.Iterable[str] | None = None,
            ignore_branches: typing.Iterable[str] | None = None,
            paths: typing.Iterable[str] | None = None,
            ignore_paths: typing.Iterable[str] | None = None,
            tags: typing.Iterable[str] | None = None,
            ignore_tags: typing.Iterable[str] | None = None,
        ) -> "_OnUpdater":
            self.branches(branches).ignore_branches(ignore_branches).paths(
                paths
            ).ignore_paths(ignore_paths).tags(tags).ignore_tags(ignore_tags)
            return on

    @dataclass
    class _Input(ProxyExpr, _IdElementUpdater[Input]):
        def __init__(self, *args, **kwargs):
            ProxyExpr.__init__(self)
            _IdElementUpdater.__init__(self, *args, **kwargs)

        def _ensure(self) -> typing.Self:
            instantated = self._instantiated
            ret = super()._ensure()
            if (
                not instantated
                and ret._path[:2] == ("on", "inputs")
                and _ctx.current_workflow
                and not _ctx.current_job
                and not ret._parent.proxied
            ):
                _ctx.error(
                    "`on.input` must be used after setting either `on.workflow_call` or `on.workflow_dispatch`"
                )
            return ret

        def __call__(
            self,
            description: str | None = None,
            *,
            id: str | None = None,
            required: bool | None = None,
            type: str | None = None,
            default: typing.Any = None,
        ) -> typing.Self:
            return (
                self.description(description)
                .required(required)
                .id(id)
                .type(type)
                .default(default)
            )

        def required(self, value: bool = True) -> typing.Self:
            return self._update("required", _value, value)

        def description(self, value: str | None) -> typing.Self:
            return self._update("description", _text, value)

        def type(self, value: str | None) -> typing.Self:
            # TODO check type
            return self._update("type", _value, value)._post_process()

        def options(self, *options: str | typing.Iterable[str]) -> typing.Self:
            return self._update("options", _seq, options)._post_process()

        def default(self, value: typing.Any) -> typing.Self:
            # TODO check type when it is already set
            return self._update("default", _value, value)._post_process()

        def _post_process(self) -> typing.Self:
            el = self._element
            try:
                el.__post_init__()
            except ValueError as e:
                _ctx.error(e.args[0])
            return self

        def _get_expr(self) -> Expr:
            if not self._instantiated:
                return ~ErrorExpr("`input` alone cannot be used in an expression")
            id = self.ensure_id()
            return getattr(Contexts.inputs, id)

    class _WorkflowDispatch(_NewUpdater):
        @property
        def input(self) -> "_OnUpdater._Input":
            return self._sub_updater(_OnUpdater._Input, "inputs", "*")

        def __call__(self) -> "_OnUpdater":
            self._ensure()
            return on

    class _WorkflowCall(_WorkflowDispatch):
        @dataclass
        class _Secret(ProxyExpr, _IdElementUpdater[Secret]):
            def __init__(self, *args, **kwargs):
                ProxyExpr.__init__(self)
                _IdElementUpdater.__init__(self, *args, **kwargs)

            def __call__(
                self,
                description: str | None = None,
                *,
                id: str | None = None,
                required: bool | None = None,
            ) -> typing.Self:
                return self.description(description).required(required).id(id)

            def required(self, value: bool = True) -> typing.Self:
                return self._update("required", _value, value)

            def description(self, value: str | None) -> typing.Self:
                return self._update("description", _text, value)

            def _get_expr(self) -> Expr:
                if not self._instantiated:
                    return ~ErrorExpr("`secret` alone cannot be used in an expression")
                id = self.ensure_id()
                return getattr(Contexts.secrets, id)

        @dataclass
        class _Output(_IdElementUpdater[Output]):
            def __call__(
                self,
                description: str | None = None,
                *,
                id: str | None = None,
                value: Value | None = None,
            ) -> typing.Self:
                return self.description(description).id(id).value(value)

            def description(self, value: str | None) -> typing.Self:
                return self._update("description", _text, value)

            def value(self, value: Value | None):
                ret = self._ensure()
                if value is not None:
                    ret.ensure_id()
                match value:
                    case Expr() | str():
                        value = str(value).replace("\0needs", "\0jobs")
                return ret._update("value", _value, value)

        @property
        def secret(self) -> _Secret:
            return self._sub_updater(self._Secret, "secrets", "*")

        @property
        def output(self) -> _Output:
            return self._sub_updater(self._Output, "outputs", "*")

    @property
    def input(self) -> _Input:
        return self._sub_updater(self._Input, "inputs", "*")

    @property
    def pull_request(self) -> PullRequest:
        return self._sub_updater(self._PullRequest, "pull_request")

    @property
    def push(self):
        return self._sub_updater(self._Push, "push")

    @property
    def workflow_dispatch(self) -> WorkflowDispatch:
        return self._sub_updater(self._WorkflowDispatch, "workflow_dispatch")

    @property
    def workflow_call(self) -> WorkflowCall:
        return self._sub_updater(self._WorkflowCall, "workflow_call")


on = _OnUpdater(_get_workflow, ("on",))


def name(name: str):
    _update_element(name, _get_job_or_workflow("name"), "name")


def env(
    mapping: ValueMapping = None,
    /,
    **kwargs: Value,
):
    _update_element(
        _map("env", (mapping, kwargs)),
        _get_job_or_workflow("env"),
        "env",
    )


class _JobUpdaters(_NewUpdater[Job]):
    def __init__(self):
        super().__init__(_get_job, ())

    def runs_on(self, runner: Value):
        self._update("runs_on", _value, runner)

    class Strategy(ProxyExpr, _NewUpdater[Strategy]):
        def __init__(self):
            ProxyExpr.__init__(self)
            _NewUpdater.__init__(self, _get_job, ("strategy",))

        def _get_expr(self) -> Expr:
            return Contexts.strategy

        class Matrix(_NewUpdater[Matrix]):
            def include(self, mapping: ValueMapping = None, /, **kwargs: Value):
                return self._update_list("includes", _field_map, (mapping, kwargs))
            def exclude(self, mapping: ValueMapping = None, /, **kwargs: Value):
                return self._update_list("excludes", _field_map, (mapping, kwargs))

            def __call__(self, value: Expr | None = None, /, includes: typing.Iterable[dict[str, Value]] | None = None, excludes: typing.Iterable[dict[str, Value]] | None = None, **kwargs: list[Value]) -> typing.Self | None:
                if value is not None:
                    if includes or excludes or kwargs:
                        _ctx.error("`matrix` cannot be used with both an expression value and `includes`, `excludes` or other keyword values")
                    self._parent.matrix = value
                    return None
                ret = self._ensure()
                ret._update("includes", _seq, includes)
                ret._update("excludes", _seq, excludes)
                ret._update("values", _field_map, (None, kwargs))
                return ret

        @property
        def matrix(self) -> Matrix:
            return self._sub_updater(self.Matrix, "matrix")

        class FailFast(ProxyExpr):
            _parent: "Strategy"

            def __init__(self, parent):
                ProxyExpr.__init__(self)
                self._parent = parent

            def _get_expr(self) -> Expr:
                return Contexts.strategy.fail_fast

            def __call__(self, value: Value | None = True) -> typing.Self:
                self._parent._update("fail_fast", _value, value)


        @property
        def fail_fast(self) -> FailFast:
            return self.FailFast(self)

        class MaxParallel(ProxyExpr):
            _parent: "Strategy"

            def __init__(self, parent):
                ProxyExpr.__init__(self)
                self._parent = parent

            def _get_expr(self) -> Expr:
                return Contexts.strategy.max_parallel

            def __call__(self, value: Value | None = None) -> typing.Self:
                self._parent._update("max_parallel", _value, value)

        @property
        def max_parallel(self) -> MaxParallel:
            return self.MaxParallel(self)

    strategy = Strategy()
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
    uses = _Updater(str)
    with_ = _Updater(dict)
    outputs = _Updater(dict)


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
                return {
                    k: _merge(k, lhs.get(k), rhs.get(k), recursed=True)
                    for k in lhs | rhs
                }
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
        _ctx.error(f"cannot assign `{rhs!r}` to `{field}`")


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

    _workflow: Workflow | None = None

    @property
    def worfklow(self) -> Workflow:
        if self._workflow is None:
            with _ctx.build_workflow(self.id) as self._workflow:
                for e in self.errors:
                    e.workflow_id = e.workflow_id or current_workflow_id()
                _ctx.errors += self.errors
                self.spec()
        return self._workflow


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
        with _ctx.build_job(id):
            func()
            return getattr(Contexts.needs, id)

    def _get_expr(self) -> Expr:
        return Contexts.job


job = _Job()


def runs_on(runner: Value):
    j = _ctx.current_job
    j_id = _ctx.current_job_id
    if j and j.uses:
        _ctx.error(
            f"job `{j_id}` cannot set `runs-on` as it has already specified `uses` (with `call`)"
        )
    else:
        _JobUpdaters.runs_on(runner)


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


# use `call` instead of `uses`, to avoid confusion with `use`
def call(target: str, **kwargs):
    j = _ctx.current_job
    j_id = _ctx.current_job_id
    if j and j.uses:
        _ctx.error(f"job `{j_id}` has already specified `uses` (with `call`)")
    elif j and j.steps:
        _ctx.error(f"job `{j_id}` specifies both `uses` (with `call`) and steps")
    elif j and j.runs_on:
        _ctx.error(f"job `{j_id}` specifies both `uses` (with `call`) and `runs-on`")
    else:
        _JobUpdaters.uses(target)
        if kwargs:
            with_(**kwargs)


def with_(*args, **kwargs):
    j = _ctx.current_job
    j_id = _ctx.current_job_id
    if j and not j.uses:
        _ctx.error(
            f"job `{j_id}` must specify `uses` (via `call`) in order to specify `with`"
        )
    else:
        kwargs = {Element._key(k): a for k, a in kwargs.items()}
        _JobUpdaters.with_(*args, **kwargs)


class _StrategyUpdater(ProxyExpr):
    def __call__(self, *args, **kwargs):
        return _JobUpdaters.strategy(*args, **kwargs)

    def _get_expr(self) -> Expr:
        return Contexts.strategy

    class FailFastUpdater(ProxyExpr):
        def __call__(self, value: Value = True):
            return _JobUpdaters.strategy.fail_fast(value)

        def _get_expr(self) -> Expr:
            return Contexts.strategy.fail_fast

    fail_fast = FailFastUpdater()

    class MaxParallelUpdater(ProxyExpr):
        def __call__(self, value: Value = True):
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


def _ensure_id(s: Step) -> str:
    if s.id is None:
        id = _get_var_name(
            lambda v: isinstance(v, _StepUpdater) and v._cached_element is s
        )
        s.id = _allocate_id(
            id or "step",
            lambda id: all(s.id != id for s in _ctx.current_job.steps),
            start_from_one=id is None,
        )
    return s.id


@dataclass
class _StepUpdater(ProxyExpr, _IdElementUpdater[Step]):
    def __init__(self, *args, **kwargs):
        ProxyExpr.__init__(self)
        _IdElementUpdater.__init__(self, *args, **kwargs)

    def __call__(
        self,
        name: Value | None = None,
        run: Value | None = None,
        id: str | None = None,
        if_: Value | None = None,
        env: dict[str, Value] | None = None,
        continue_on_error: bool | None = None,
        uses: str | None = None,
        with_: dict[str, Value] | None = None,
        outputs: tuple[str] | None = None,
        needs: typing.Iterable[RefExpr] | None = None,
    ) -> typing.Self:
        ret = (
            self.name(name)
            .id(id)
            .if_(if_)
            .continue_on_error(continue_on_error)
            .needs(needs)
            .env(env)
        )
        if run is not None:
            ret.run(run)
        if uses is not None:
            ret.uses(uses)
        if with_ is not None:
            ret.with_(with_)
        if outputs is not None:
            ret.outputs(outputs)
        return ret

    def _ensure(self):
        if (
            _ctx.current_job
            and _ctx.current_job.uses is not None
            and not _ctx.current_job.steps
        ):
            _ctx.error(
                f"job `{_ctx.current_job_id}` adds steps when `uses` is already set (by `call`)"
            )
        return super()._ensure()

    def _get_expr(self) -> Expr:
        if not self._instantiated:
            return ~ErrorExpr("`step` alone cannot be used in an expression")
        id = self.ensure_id()
        return getattr(Contexts.steps, id)

    def _ensure_run_step(self) -> typing.Self:
        ret = self._ensure()
        if ret._element.uses is not None or ret._element.with_:
            _ctx.error("cannot turn a `uses` step into a `run` one")
        return ret

    def _ensure_use_step(self) -> typing.Self:
        ret = self._ensure()
        if ret._element.run is not None:
            _ctx.error("cannot turn a `run` step into a `uses` one")
        return ret

    def name(self, name: Value) -> typing.Self:
        return self._update("name", _value, name)

    def if_(self, condition: Value) -> typing.Self:
        return self._update("if_", _value, condition)

    def env(
        self,
        mapping: ValueMapping = None,
        /,
        **kwargs: Value,
    ) -> typing.Self:
        return self._update("env", _map, (mapping, kwargs))

    def run(self, code: Value) -> typing.Self:
        return self._ensure_run_step()._update("run", _text, code)

    def uses(self, source: Value, **kwargs: Value):
        ret = self._ensure_run_step()._update("uses", _value, source)
        if isinstance(source, str) and not ret._element.name:
            try:
                _, _, action_name = source.rpartition("/")
                action_name, _, _ = action_name.partition("@")
                action_name = inflection.humanize(inflection.titleize(action_name))
                ret.name(action_name)
            except Exception:
                pass
        if kwargs:
            ret.with_(**kwargs)
        return ret

    def continue_on_error(self, value: Value = True) -> typing.Self:
        return self._update("continue_on_error", _value, value)

    def with_(
        self,
        mapping: ValueMapping = None,
        /,
        **kwargs: Value,
    ) -> typing.Self:
        return self._ensure_use_step()._update("with_", _field_map, (mapping, kwargs))

    class _StepOutputs(ProxyExpr):
        _parent: "_StepUpdater"

        def __init__(self, parent: "_StepUpdater"):
            super().__init__()
            self._parent = parent

        def __call__(self, *args: str, **kwargs: Value) -> "_StepUpdater":
            args += tuple(a for a in kwargs if a not in args)
            ret = self._parent._update("outputs", _seq, args)
            # TODO: support other shells than bash
            if kwargs:
                # TODO: handle quoting?
                out_code = "\n".join(
                    f"echo {k}={v} >> $GITHUB_OUTPUTS" for k, v in kwargs.items()
                )
                previous_code = ret._element.run
                # validate addition without repeating errors on existing code
                ret.run(out_code)
                ret.run(
                    f"{previous_code}\n{out_code}"
                    if previous_code is not None
                    else out_code
                )
            return ret

        def _get_expr(self) -> Expr:
            return self._parent._get_expr().outputs

    @property
    def outputs(self) -> _StepOutputs:
        return self._StepOutputs(self)

    def needs(self, *jobs: RefExpr | tuple[RefExpr, ...]) -> typing.Self:
        jobs = _seq(self._log_path, jobs)
        if jobs is not None:
            jobs = needs(*jobs)
        return self._update("needs", _value, jobs)


step = _StepUpdater(_get_job, ("steps", "*"))
run = step.run
use = step.uses


def _dump_step_outputs(s: Step):
    id = _ensure_id(s)
    if s.outputs:
        outputs = getattr(steps, id).outputs
        _JobUpdaters.outputs((o, getattr(outputs, o)) for o in s.outputs)
    else:
        _ctx.error(
            f"step `{id}` passed to `outputs`, but no outputs were declared on it. Use `returns()` to do so",
        )


def _dump_job_outputs(id: str, j: Job):
    if j.outputs:
        outputs = getattr(Contexts.jobs, id).outputs
        on.workflow_call._outputs(
            (o, Output(value=getattr(outputs, o))) for o in j.outputs
        )
    else:
        _ctx.error(
            f"job `{id}` passed to `outputs`, but no outputs were declared on it. Use `outputs()` to do so",
        )


def outputs(*args: RefExpr | _StepUpdater, **kwargs: typing.Any):
    for arg in args:
        match arg:
            case RefExpr(_segments=(*_, id)):
                _JobUpdaters.outputs(((id, arg),))
            case _StepUpdater():
                _dump_step_outputs(arg._element)
            case _:
                _ctx.error(
                    f"unsupported unnamed output `{instantiate(arg)}`, must be a context field or a step"
                )
    if kwargs:
        _JobUpdaters.outputs(**{Element._key(k): a for k, a in kwargs.items()})


always = function("always", 0)
cancelled = function("cancelled", 0)
fromJson = function("fromJson")
contains = function("contains", 2)
