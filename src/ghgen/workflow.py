import dataclasses
import typing

from ruamel.yaml import CommentedSeq

from .element import Element
from typing import Any, cast
from .expr import Value, Expr, ProxyExpr, RefExpr, instantiate, ErrorExpr
from dataclasses import field
from ruamel.yaml.scalarstring import LiteralScalarString
from ruamel.yaml.comments import CommentedMap, CommentedSeq


def _set_flow_style(d: dict, *fields) -> dict:
    for f in fields:
        if f not in d:
            continue
        match d[f]:
            case list():
                d[f] = CommentedSeq(d[f])
            case dict():
                d[f] = CommentedMap(d[f])
            case _:
                continue
        d[f].fa.set_flow_style()
    return d


class Input[T](Element):
    Type: typing.ClassVar[type] = typing.Literal[
        "boolean", "choice", "number", "environment", "string"
    ]

    description: str
    _: dataclasses.KW_ONLY
    id: str
    required: bool = False
    default: T
    type: Type = "string"
    options: list[str]

    def asdict(self) -> typing.Any:
        ret = super().asdict()
        ret.pop("id", None)
        return ret

    def __post_init__(self):
        if self.default is not None:
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
        elif self.type not in (None,) + tuple(typing.get_args(self.Type)):
            raise ValueError(f"unexpected input type `{self.type}`")
        if self.options:
            self.type = "choice"


class Secret(Element):
    description: str
    _: dataclasses.KW_ONLY
    required: bool = False


class Trigger(Element):
    types: list[str]


class TypedTrigger(type):
    def __new__(cls, *types: str):
        return type(
            "TypedTrigger",
            (Trigger,),
            {
                "allowed_types": tuple(sorted(types)),
            },
        )


class ChangeTrigger(Element):
    branches: list[str]
    ignore_branches: list[str]
    paths: list[str]
    ignore_paths: list[str]


class PullRequest(
    ChangeTrigger,
    TypedTrigger(
        "assigned",
        "unassigned",
        "labeled",
        "unlabeled",
        "opened",
        "edited",
        "closed",
        "reopened",
        "synchronize",
        "converted_to_draft",
        "locked",
        "unlocked",
        "enqueued",
        "dequeued",
        "milestoned",
        "demilestoned",
        "ready_for_review",
        "review_requested",
        "review_request_removed",
        "auto_merge_enabled",
        "auto_merge_disabled",
    ),
):
    pass


class PullRequestTarget(
    ChangeTrigger,
    TypedTrigger(
        "assigned",
        "unassigned",
        "labeled",
        "unlabeled",
        "opened",
        "edited",
        "closed",
        "reopened",
        "synchronize",
        "converted_to_draft",
        "ready_for_review",
        "locked",
        "unlocked",
        "review_requested",
        "review_request_removed",
        "auto_merge_enabled",
        "auto_merge_disabled",
    ),
):
    pass


class Push(ChangeTrigger):
    tags: list[str]
    ignore_tags: list[str]


class Schedule(Element):
    cron: str


def _dictionarize_inputs(inputs: list[Input], d: dict) -> dict:
    if inputs is None:
        return d
    serialized: list[Input] = d.pop("inputs")
    d["inputs"] = {s.id: i for s, i in zip(inputs, serialized) if s.id}
    return d


class WorkflowDispatch(Element):
    inputs: list[Input]

    def asdict(self) -> typing.Any:
        return _dictionarize_inputs(self.inputs, super().asdict())


class WorkflowCall(Element):
    inputs: list[Input]
    secrets: dict[str, Secret]

    def asdict(self) -> typing.Any:
        return _dictionarize_inputs(self.inputs, super().asdict())


class On(Element):
    _preserve_underscores = True

    branch_protection_rule: TypedTrigger("created", "edited", "deleted")
    check_run: TypedTrigger("created", "completed", "requested_action", "rerequested")
    check_suite: TypedTrigger("completed")
    create: Element
    delete: Element
    deployment: Element
    deployment_status: Element
    discussion: TypedTrigger(
        "created",
        "edited",
        "deleted",
        "transferred",
        "pinned",
        "unpinned",
        "labeled",
        "unlabeled",
        "locked",
        "unlocked",
        "category_changed",
        "answered",
        "unanswered",
    )
    discussion_comment: TypedTrigger("created", "edited", "deleted")
    fork: Element
    gollum: Element
    issue_comment: TypedTrigger("created", "edited", "deleted")
    issues: TypedTrigger(
        "opened",
        "edited",
        "deleted",
        "transferred",
        "pinned",
        "unpinned",
        "closed",
        "reopened",
        "assigned",
        "unassigned",
        "labeled",
        "unlabeled",
        "locked",
        "unlocked",
        "milestoned",
        "demilestoned",
    )
    label: TypedTrigger("created", "edited", "deleted")
    merge_group: TypedTrigger("checks_requested")
    milestone: TypedTrigger("created", "closed", "opened", "edited", "deleted")
    page_build: Element
    public: Element
    pull_request: PullRequest
    pull_request_review: TypedTrigger("submitted", "edited", "dismissed")
    pull_request_review_comment: TypedTrigger("created", "edited", "deleted")
    pull_request_target: PullRequestTarget
    push: Push
    registry_package: TypedTrigger("published", "updated")
    release: TypedTrigger(
        "published",
        "unpublished",
        "created",
        "edited",
        "deleted",
        "prereleased",
        "released",
    )
    repository_dispatch: Trigger
    schedule: Schedule
    status: Element
    watch: TypedTrigger("started")
    workflow_call: WorkflowCall
    workflow_dispatch: WorkflowDispatch
    workflow_run: TypedTrigger("completed", "in_progress", "requested")


class Step(Element):
    id: str
    name: Value
    if_: Value
    continue_on_error: Value
    run: Value
    env: dict[str, Value]
    uses: str
    with_: dict[str, Value]

    # extensions
    outputs: list[str]
    needs: list[str]

    def asdict(self) -> typing.Any:
        ret = super().asdict()
        ret.pop("outputs", None)
        needs = ret.pop("needs", None)
        if isinstance(self.if_, Expr):
            ret["if"] = self.if_._formula
        run = ret.get("run")
        if run and "\n" in run:
            if run[-1] != "\n":
                run += "\n"
            ret["run"] = LiteralScalarString(run)
        if needs:
            ret = CommentedMap(ret)
            ret.yaml_set_start_comment(f"needs {", ".join(needs)}", indent=4)
        return ret


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
        values = ret.pop("values", {})
        ret |= values
        return _set_flow_style(ret, *values)


class Strategy(Element):
    matrix: Matrix
    fail_fast: Value
    max_parallel: Value


class Credentials(Element):
    username: Value
    password: Value


class Container(Element):
    image: Value
    _: dataclasses.KW_ONLY
    credentials: Credentials
    env: dict[str, Value]
    ports: list[Value]
    volumes: list[Value]
    options: list[Value]


default_runner = "ubuntu-latest"


class Job(Element):
    name: str
    needs: list[str]
    runs_on: str
    container: Container
    services: dict[str, Container]
    outputs: dict[str, Value]
    strategy: Strategy
    env: dict[str, Value]
    steps: list[Step]
    uses: str
    with_: dict[str, Value]

    def asdict(self) -> typing.Any:
        return _set_flow_style(super().asdict(), "needs")


class Workflow(Element):
    name: str
    on: On = field(default_factory=On)
    outputs: dict[str, Value]
    env: dict[str, Value]
    jobs: dict[str, Job] = field(default_factory=dict)

    @property
    def inputs(self) -> list[Input]:
        inputs = {
            id(i): i
            for t in (self.on.workflow_call, self.on.workflow_dispatch)
            if t is not None
            for i in t.inputs or ()
        }
        inputs = {
            id(i): i
            for t in (self.on.workflow_call, self.on.workflow_dispatch)
            if t is not None
            for i in t.inputs or ()
        }
        return [*inputs.values()]
