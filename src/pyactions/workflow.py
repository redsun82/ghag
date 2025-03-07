import dataclasses
import typing

from .element import element, Element
from typing import ClassVar, Any, cast
from .expr import Value, Expr
from dataclasses import field

__all__ = [
    "PullRequest",
    "WorkflowDispatch",
    "On",
    "Step",
    "RunStep",
    "UseStep",
    "Job",
    "Strategy",
    "Matrix",
    "Workflow",
    "Input",
    "Secret",
    "InputProxy",
    "Choice",
]


@element
class Input[T]:
    description: str
    required: bool = False
    default: T
    type: typing.Literal["boolean", "choice", "number", "environment", "string"] = (
        "string"
    )
    options: list[str]

    def __post_init__(self):
        if self.type is None and self.default is not None:
            self.type = type(self.default)
        if self.type is bool:
            self.type = "boolean"
        elif self.type in (int, float):
            self.type = "number"
        elif self.type is str:
            self.type = "string"
        elif typing.get_origin(self.type) is typing.Literal:
            self.options = list(typing.get_args(self.type))
            self.type = "choice"
        elif (typing.get_origin(self.type) or self.type) is dict:
            self.type = "environment"


type Choice[*Args] = Input[typing.Literal[*Args]]


@dataclasses.dataclass
class InputProxy(Expr):
    proxied: list[Input] = dataclasses.field(default_factory=list)

    def __init__(self, key: str, *proxied: Input):
        super().__init__(f"inputs.{key}")
        self.proxied = list(proxied)

    def __setattr__(self, name, value):
        if any(f.name == name for f in dataclasses.fields(Input)):
            for p in self.proxied:
                setattr(p, name, value)
        else:
            super().__setattr__(name, value)


@element
class Secret:
    description: str
    required: bool = False


@element
class PullRequest:
    branches: list[str]
    paths: list[str]


@element
class WorkflowDispatch:
    inputs: dict[str, Input]


@element
class WorkflowCall:
    inputs: dict[str, Input]
    secrets: dict[str, Secret]
    # TODO outputs


@element
class On:
    workflow_call: WorkflowCall
    workflow_dispatch: WorkflowDispatch
    pull_request: PullRequest


@element
class Step:
    id: str
    name: Value[str]
    if_: Value[bool]
    continue_on_error: Value[bool]


@element
class RunStep(Step):
    run: Value[str]
    env: dict[str, Value[str]]


@element
class UseStep(Step):
    use: str
    with_: dict[str, Value[str | bool | int | float]]


@element
class Run(Step):
    run: str
    shell: str
    working_directory: str


@element
class Use(Step):
    use: str
    with_: dict[str, str]


@element
class Matrix:
    include: list[dict[str, str]]
    exclude: list[dict[str, str]]
    values: dict[str, list[str]]

    def __init__(
        self,
        *,
        include: list[dict[str, str]] = None,
        exclude: list[dict[str, str]] = None,
        **values: list[str],
    ):
        self.include = include
        self.exclude = exclude
        self.values = values

    def asdict(self) -> dict[str, Any]:
        ret = Element.asdict(cast(Element, self))
        ret |= ret.pop("values", {})
        return ret


@element
class Strategy:
    matrix: Matrix
    fail_fast: Value[bool]
    max_parallel: Value[int]


@element
class Job:
    name: str
    runs_on: str = "ubuntu-latest"
    strategy: Strategy
    env: dict[str, Value]
    steps: list[Step]

    def step_by_id(self, id: str) -> Step | None:
        return next((s for s in self.steps if s.id == id), None)


@element
class Workflow:
    name: str
    on: On = field(default_factory=On)
    env: dict[str, Value]
    jobs: dict[str, Job] = field(default_factory=dict)
