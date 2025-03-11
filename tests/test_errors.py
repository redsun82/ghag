import inspect

import pytest

from conftest import expect_errors
from src.pyactions.ctx import *


@expect_errors
def test_wrong_types(error):
    on.pull_request(branches=["main"])
    error("cannot assign `str` to `branches`")
    on.pull_request(branches="dev")
    error("cannot assign `int` to `env`")
    env(3)
    env(FOO="bar")
    error("cannot assign `list` to `env`")
    env(["no"])
    error("cannot assign `bool` to `runs_on`")
    runs_on(True)


@expect_errors
def test_wrong_jobs(error):
    on.workflow_dispatch()

    @job
    def external():
        run("echo Hello, world")

        # fmt: off
        error("job `nested` not created directly inside a workflow body")
        @job
        # fmt: on
        def nested():
            run("false")

    @job
    def another():
        pass

    # fmt: off
    error("job `external` already exists in workflow `test_wrong_jobs`")
    @job
    # fmt: on
    def external():
        run("nope")


def test_job_outside_workflow():
    with pytest.raises(GenerationError) as e:

        @job
        def no():
            pass

    assert e.value.errors == [
        Error(
            __file__,
            lineno=inspect.getsourcelines(test_job_outside_workflow)[1] + 3,
            workflow_id=None,
            message="job `no` not created directly inside a workflow body",
        ),
    ]


@expect_errors
def test_auto_job_with_existing_jobs(error):
    @job
    def a_job():
        pass

    error(
        "`runs_on` is a `job` field, but implicit job cannot be created because there are already jobs in the workflow"
    )
    runs_on("x")


@expect_errors
def test_adding_jobs_to_auto_job(error):
    runs_on("x")

    # fmt: off
    error("explict job `a_job` cannot be created after already implicitly creating a job, which happened when setting `runs_on`")
    @job
    # fmt: on
    def a_job():
        pass


@expect_errors
def test_workflow_fields_in_job(error):
    @job
    def a_job():
        name("a name")

        error("`on` is not a job field")
        on.workflow_dispatch()


@expect_errors
def test_workflow_fields_in_auto_job(error):
    runs_on("x")

    error(
        "`on` is not a job field, and an implicit job was created when setting `runs_on`"
    )
    on.workflow_dispatch()


def test_wrong_annotation():
    @workflow
    def wf(x: None, y: list[int]):
        pass

    with pytest.raises(GenerationError) as e:
        wf.instantiate()
    assert e.value.errors == [
        Error(
            __file__,
            lineno=inspect.getsourcelines(test_wrong_annotation)[1] + 1,
            workflow_id="wf",
            message=f"unexpected input type `{t}` for workflow parameter `{p}`",
        )
        for p, t in (("x", "None"), ("y", "list[int]"))
    ]


@expect_errors
def test_wrong_outputs(error):
    # fmt: off
    error("job `j1` returns expression `steps.x.outputs` which has no declared fields. Did you forget to use `returns()` on a step?")
    @job
    def j1():
        x = step("x")
        return x.outputs

    error("job `j2` returns expression `steps.y.outputs` which has no declared fields. Did you forget to use `returns()` on a step?")
    @job
    def j2():
        x = step("x").returns("foo")
        y = step("y")
        return x.outputs, y.outputs

    error("unsupported return value for job `j3`, must be `None`, a dictionary, a step `outputs` or a tuple of step `outputs`")
    @job
    def j3():
        x = step("x").returns("foo")
        y = step("y")
        return x.outputs, y

    error("unsupported return value for job `j4`, must be `None`, a dictionary, a step `outputs` or a tuple of step `outputs`")
    @job
    def j4():
        return 42
    # fmt: on
