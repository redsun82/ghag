import sys
import typing
from dataclasses import dataclass, field, fields
from typing import ClassVar, Self


def _process(x: typing.Any):
    match x:
        case dict():
            return {
                i.rstrip("_").replace("_", "-"): _process(v)
                for i, v in x.items()
                if v is not None
            }
        case list():
            return [_process(v) for v in x]
        case _:
            return x


def _clean_code(code: str):
    lines = code.splitlines()
    assert lines
    indent = sys.maxsize
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    trimmed = [lines[0].strip()]
    if indent < sys.maxsize:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)
    return "\n".join(trimmed)


def _string_literal(s: str) -> str:
    return f"'{s.replace("'", "''")}'"


@dataclass
class Expr:
    _value: str
    _op: str | None = None

    def _as_and_operand(self) -> str:
        return f"({self._value})" if self._op == "||" else self._value

    @classmethod
    def _syntax(cls, v: "ExprOrLiteral", within_and: bool = False) -> str:
        match v:
            case Expr() as e if within_and:
                return e._as_and_operand()
            case Expr():
                return e._value
            case str() as s:
                return _string_literal(s)
            case _:
                return str(v)

    def __str__(self):
        return f"${{{{ {self._value} }}}}"

    def __and__(self, other: "ExprOrLiteral") -> Self:
        return Expr(f"{self._as_and_operand()} && {self._syntax(other, True)}", "&&")

    def __rand__(self, other: "ExprOrLiteral") -> Self:
        return Expr(f"{self._syntax(other, True)} && {self._as_and_operand()}", "&&")

    def __or__(self, other: "ExprOrLiteral") -> Self:
        return Expr(f"{self._value} || {self._syntax(other)}", "||")

    def __ror__(self, other: "ExprOrLiteral") -> Self:
        return Expr(f"{self._syntax(other)} || {self._value}", "||")

    def __invert__(self) -> Self:
        operand = f"({self._value})" if self._op else self._value
        return Expr(f"!{operand}")


ExprOrLiteral = Expr | str | int | float | bool


@dataclass
class Root:
    def asdict(self) -> dict[str, typing.Any]:
        return _process(asdict(self))


@dataclass
class Trigger(Root):
    tag: ClassVar[str]


@dataclass
class PullRequest(Trigger):
    tag: ClassVar[str] = "pull_request"
    branches: list[str] | None = None
    paths: list[str] | None = None


@dataclass
class WorkflowDispatch(Trigger):
    tag: ClassVar[str] = "workflow_dispatch"


@dataclass
class On(Root):
    pull_request: PullRequest | None = None
    workflow_dispatch: WorkflowDispatch | None = None


@dataclass
class Step(Root):
    name: str | None = None
    if_: str | None = None
    env: dict[str, str] | None = None
    continue_on_error: str | bool | None = None


@dataclass
class Run(Step):
    run: str | None = None
    shell: str | None = None
    working_directory: str | None = None


@dataclass
class Use(Step):
    use: str | None = None
    with_: dict[str, str] | None = None


@dataclass
class Matrix(Root):
    include: list[dict[str, str]] | None = None
    exclude: list[dict[str, str]] | None = None
    values: dict[str, list[str]] | None = None

    def asdict(self) -> dict[str, typing.Any]:
        print("PROUT")
        ret = super().asdict()
        ret |= ret.pop("values", {})
        return ret


@dataclass
class Strategy(Root):
    matrix: Matrix | None = None
    fail_fast: ExprOrLiteral | None = None
    max_parallel: ExprOrLiteral | None = None


@dataclass
class Job(Root):
    name: str | None = None
    runs_on: str | None = None
    strategy: Strategy | None = None
    steps: list[Step] | None = None


@dataclass
class Workflow(Root):
    name: str | None = None
    on: On = field(default_factory=On)
    jobs: dict[str, Job] = field(default_factory=dict)


@dataclass
class _Ctx:
    class On:
        def pull_request(self, **kwargs) -> Self:
            assert isinstance(_ctx.current, Workflow)
            _ctx.current.on.pull_request = PullRequest(**kwargs)
            return self

        def workflow_dispatch(self, **kwargs) -> Self:
            assert isinstance(_ctx.current, Workflow)
            _ctx.current.on.workflow_dispatch = WorkflowDispatch(**kwargs)
            return self

    @dataclass
    class StepBuilder:
        building: Step = field(default_factory=Step)

        def name(self, n: Expr) -> Self:
            assert self.building.name is None
            self.building.name = n
            return self

        def if_(self, cond: ExprOrLiteral) -> Self:
            assert self.building.if_ is None
            self.building.if_ = str(cond)
            return self

        def env(self, env: dict[str, ExprOrLiteral]) -> Self:
            if self.building.env is None:
                self.building.env = {}
            self.building.env.update((k, str(v)) for k, v in env.items())
            return self

        def run(self, code: ExprOrLiteral) -> Self:
            match self.building:
                case Run() as r:
                    assert r.run is None
                    r.run = str(code)
                case Use():
                    assert False
                case _:
                    self.building = Run(
                        run=_clean_code(str(code)), **asdict(self.building)
                    )
            return self

        def use(self, action: str) -> Self:
            match self.building:
                case Use() as u:
                    assert u.use is None
                    u.use = action
                case Run():
                    assert False
                case _:
                    self.building = Use(use=action, **asdict(self.building))
            return self

        def working_directory(self, d: ExprOrLiteral) -> Self:
            match self.building:
                case Run() as r:
                    assert r.working_directory is None
                    r.working_directory = str(d)
                case Use():
                    assert False
                case _:
                    self.building = Run(
                        working_directory=str(d), **asdict(self.building)
                    )
            return self

        def with_(self, args: dict[str, ExprOrLiteral]) -> Self:
            match self.building:
                case Use() as u:
                    if u.with_ is None:
                        u.with_ = {}
                    u.with_.update((k, str(v)) for k, v in args.items())
                case Run():
                    assert False
                case _:
                    self.building = Use(with_=args, **asdict(self.building))
            return self

    class Step:
        def __getattr__(self, item) -> typing.Callable:
            if item.startswith("_"):
                raise AttributeError(item)
            j = _ctx.current
            assert isinstance(j, Job)
            if j.steps is None:
                j.steps = []
            if _ctx.step is not None:
                j.steps.append(_ctx.step.building)
            _ctx.step = _Ctx.StepBuilder()
            ret = getattr(_ctx.step, item)
            assert callable(ret)
            return ret

        def __call__(self, n: str) -> "_Ctx.StepBuilder":
            return self.name(n)

    current: object = None
    workflows: dict[str, Workflow] = field(default_factory=dict)
    step: None | StepBuilder = None


_ctx = _Ctx()

on = _Ctx.On()
step = _Ctx.Step()


def failed() -> Expr:
    return Expr("failed()")


def skipped() -> Expr:
    return Expr("skipped()")


def workflow(f):
    _ctx.current = _ctx.workflows[f.__name__] = Workflow()
    f()
    _ctx.current = None


def job(f):
    assert isinstance(_ctx.current, Workflow)
    wf = _ctx.current
    j = _ctx.current = wf.jobs[f.__name__] = Job()
    f()
    if _ctx.step is not None:
        j.steps.append(_ctx.step.building)
    _ctx.current = wf


def runs_on(runner: Expr):
    assert isinstance(_ctx.current, Job)
    assert _ctx.current.runs_on is None
    _ctx.current.runs_on = runner


def matrix(
    include: list[dict[str, ExprOrLiteral]] | None = None,
    exclude: list[dict[str, ExprOrLiteral]] | None = None,
    **kwargs: Expr | list[ExprOrLiteral],
):
    assert isinstance(_ctx.current, Job)
    m = Matrix(
        {k: str(v) for k, v in include.items()} if include else None,
        {k: str(v) for k, v in include.items()} if exclude else None,
        {k: str(v) for k, v in kwargs.items()} if kwargs else None,
    )
    if _ctx.current.strategy is None:
        _ctx.current.strategy = Strategy(m)
    else:
        assert _ctx.current.strategy.matrix is None
        _ctx.current.strategy.matrix = m
