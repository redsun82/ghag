import dataclasses
import typing

from .element import Element
from typing import Any, cast
from .expr import Value, Expr
from dataclasses import field
from ruamel.yaml.scalarstring import LiteralScalarString
from ruamel.yaml.comments import CommentedMap


class Input[T](Element):
    Type: typing.ClassVar[type] = typing.Literal[
        "boolean", "choice", "number", "environment", "string"
    ]

    description: str
    _: dataclasses.KW_ONLY
    required: bool = False
    default: T
    type: Type = "string"
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
        elif self.type not in (None,) + tuple(typing.get_args(self.Type)):
            raise ValueError(f"unexpected input type `{self.type}`")


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


class WorkflowDispatch(Element):
    inputs: dict[str, Input]


class WorkflowCall(Element):
    inputs: dict[str, Input]
    secrets: dict[str, Secret]
    # TODO outputs


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
    name: Value[str]
    if_: Value[bool]
    continue_on_error: Value[bool]

    outputs: list[str]

    def asdict(self) -> typing.Any:
        ret = super().asdict()
        ret.pop("outputs", None)
        if isinstance(self.if_, Expr):
            ret["if"] = self.if_._value
        ret = CommentedMap(ret)
        ret.fa.set_block_style()
        return ret


class RunStep(Step):
    run: Value[str]
    env: dict[str, Value[str]]

    def asdict(self) -> typing.Any:
        ret = super().asdict()
        run = ret.get("run")
        if run and "\n" in run:
            if run[-1] != "\n":
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


class Credentials(Element):
    username: Value[str]
    password: Value[str]


class Container(Element):
    image: Value[str]
    _: dataclasses.KW_ONLY
    credentials: Credentials
    env: dict[str, Value[str]]
    ports: list[Value[int]]
    volumes: list[Value[str]]
    options: list[Value[str]]


class Job(Element):
    name: str
    needs: list[str]
    runs_on: str = "ubuntu-latest"
    container: Container
    services: dict[str, Container]
    outputs: dict[str, Value[str]]
    strategy: Strategy
    env: dict[str, Value[str]]
    steps: list[Step]

    def asdict(self) -> typing.Any:
        ret = super().asdict()
        outputs = ret.get("outputs")
        if outputs:
            outputs = CommentedMap(outputs)
            outputs.fa.set_block_style()
            ret["outputs"] = outputs
        return ret


class Workflow(Element):
    name: str
    on: On = field(default_factory=On)
    env: dict[str, Value]
    jobs: dict[str, Job] = field(default_factory=dict)
