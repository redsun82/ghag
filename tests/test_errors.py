import inspect

import pytest

from conftest import expect_errors
from src.ghag.ctx import *


@expect_errors
def test_wrong_types(error):
    on.pull_request(branches=["main"])
    error("cannot assign `str` to `branches`")
    on.pull_request(branches="dev")
    error("illegal assignment to `env`")
    env(3)
    env(FOO="bar")
    error("illegal assignment to `env`")
    env(["nope"])


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

        error("`on` is a workflow field, it cannot be set in job `a_job`")
        on.workflow_dispatch()


@expect_errors
def test_workflow_fields_in_auto_job(error):
    runs_on("x")

    error(
        "`on` is a workflow field, and an implicit job was created when setting `runs_on`"
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
def test_unexpected_step_outputs(error):
    x = step("x")
    error("`foo` was not declared in step `x`, use `returns()` declare it")
    step("y").run(x.outputs.foo)


@expect_errors
def test_wrong_outputs(error):
    # fmt: off
    # error("job `j1` returns expression `steps.x.outputs` which has no declared fields. Did you forget to use `returns()` on a step?")
    # @job
    # def j1():
    #     x = step("x")
    #     return x.outputs

    error("job `y` returns expression `steps.y.outputs` which has no declared fields. Did you forget to use `returns()` on a step?")
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


@expect_errors
def test_undeclared_step_output(error):
    x = step("step1").returns("foo")
    error("`bar` was not declared in step `x`, use `returns()` declare it")
    step("step2").run(x.outputs.bar)


@expect_errors
def test_wrong_job_needs(error):
    # fmt: off
    error("job `j1` needs job `non_existent` which is currently undefined")
    @job
    def j1(non_existent):
        pass

    error("job `j2` needs job `j3` which is currently undefined")
    @job
    def j2(j3):
        pass

    error("job `j3` needs job `wat` which is currently undefined")
    @job
    def j3(j1, wat, j2):
        pass

    @job
    def j4():
        error("job `j3` is not a prerequisite, you must add it to `j3`'s parameters")
        run(j3.outputs)
    # fmt: on


@expect_errors
def test_unavailable_job_contexts(error):
    _ = str(matrix)
    _ = str(steps)
    _ = str(job)

    @job
    def j():
        error("`matrix` can only be used in a matrix job")
        step(matrix.x)


@expect_errors
def test_unavailable_container(error):
    @job
    def j1():
        error("`job.container` can only be used in a containerized job")
        step(job.container.id)


@expect_errors
def test_unavailable_service(error):
    @job
    def j1():
        error("`job.services` can only be used in a job with services")
        step(job.services)

    @job
    def j2():
        service("a")
        error("no `b` service defined in `job.services`")
        step(job.services.b)


@expect_errors
def test_unavailable_matrix_values(error):
    @job
    def j1():
        strategy.matrix(a=[0])
        error("`x` was not declared in the `matrix` for this job")
        step(matrix.x)

    @job
    def j2():
        strategy.matrix(b=["x"])
        error("`a` was not declared in the `matrix` for this job")
        step(matrix.a)

    @job
    def j3():
        strategy.matrix(fromJson("{}ß"))

    @job
    def j4():
        strategy.matrix(x=[42])
        error("`a` was not declared in the `matrix` for this job")
        step(matrix.a)
