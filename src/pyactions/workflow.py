from .element import element, Element
from typing import ClassVar, Any, cast
from .expr import Value
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
]


@element
class PullRequest:
    tag: ClassVar[str] = "pull_request"
    branches: list[str]
    paths: list[str]


@element
class WorkflowDispatch:
    tag: ClassVar[str] = "workflow_dispatch"


@element
class On:
    pull_request: PullRequest
    workflow_dispatch: WorkflowDispatch


@element
class Step:
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


@element
class Workflow:
    name: str
    on: On = field(default_factory=On)
    env: dict[str, Value]
    jobs: dict[str, Job] = field(default_factory=dict)
