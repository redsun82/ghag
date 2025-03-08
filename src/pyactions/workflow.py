import dataclasses
import typing

from .element import Element
from typing import Any, cast
from .expr import Value, Expr
from dataclasses import field
from ruamel.yaml.scalarstring import LiteralScalarString

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


class Input[T](Element):
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


class Secret(Element):
    description: str
    required: bool = False


class PullRequest(Element):
    branches: list[str]
    paths: list[str]


class WorkflowDispatch(Element):
    inputs: dict[str, Input]


class WorkflowCall(Element):
    inputs: dict[str, Input]
    secrets: dict[str, Secret]
    # TODO outputs


class On(Element):
    workflow_call: WorkflowCall
    workflow_dispatch: WorkflowDispatch
    pull_request: PullRequest


class Step(Element):
    id: str
    name: Value[str]
    if_: Value[bool]
    continue_on_error: Value[bool]

    def asdict(self) -> typing.Any:
        ret = super().asdict()
        if isinstance(self.if_, Expr):
            ret["if"] = self.if_._value
        return ret


class RunStep(Step):
    run: Value[str]
    env: dict[str, Value[str]]

    def asdict(self) -> typing.Any:
        ret = super().asdict()
        run = ret["run"]
        if run and run[-1] != "\n":
            run += "\n"
        ret["run"] = LiteralScalarString(run)
        return ret


class UseStep(Step):
    use: str
    with_: dict[str, Value[str | bool | int | float]]


class Matrix(Element):
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


class Strategy(Element):
    matrix: Matrix
    fail_fast: Value[bool]
    max_parallel: Value[int]


class Job(Element):
    name: str
    runs_on: str = "ubuntu-latest"
    strategy: Strategy
    env: dict[str, Value]
    steps: list[Step]

    def step_by_id(self, id: str) -> Step | None:
        return next((s for s in self.steps if s.id == id), None)


class Workflow(Element):
    name: str
    on: On = field(default_factory=On)
    env: dict[str, Value]
    jobs: dict[str, Job] = field(default_factory=dict)
