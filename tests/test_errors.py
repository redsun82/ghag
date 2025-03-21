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
        error('job `nested` not created directly inside a workflow body')
        @job
        # fmt: on
        def nested():
            run("false")

    @job
    def another():
        pass

    # fmt: off
    error('job `external` already exists in workflow `test_wrong_jobs`')
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
    error('explict job `a_job` cannot be created after already implicitly creating a job, which happened when setting `runs_on`')
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
    @job
    def j1():
        x = step("x")
        error(
            "step `x` passed to `outputs`, but no outputs were declared on it. Use `returns()` to do so"
        )
        outputs(x)

    @job
    def j2():
        x = step("x").returns("foo")
        y = step("y")
        error(
            "step `y` passed to `outputs`, but no outputs were declared on it. Use `returns()` to do so"
        )
        outputs(x, y)

    @job
    def j3():
        error(
            'unsupported unnamed output `42`, must be `"*"`, a context field or a step'
        )
        outputs(42)

    @job
    def j4():
        step.id("x").returns("foo")
        step.id("y").returns("bar")
        error(
            'unsupported unnamed output `${{ steps.x && steps.y }}`, must be `"*"`, a context field or a step'
        )
        outputs(steps.x & steps.y)

    error(
        "job `j3` passed to `outputs`, but no outputs were declared on it. Use `outputs()` to do so"
    )
    outputs(j3)


@expect_errors
def test_undeclared_step_output(error):
    x = step("step1").returns("foo")
    error("`bar` was not declared in step `x`, use `returns()` declare it")
    step("step2").run(x.outputs.bar)


@expect_errors
def test_wrong_job_needs(error):
    @job
    def init():
        pass

    @job
    def j():
        error("no `non_existing` job declared yet in this workflow")
        needs(Contexts.needs.non_existing)
        error("no `other_job` job declared yet in this workflow")
        needs(Contexts.needs.other_job)
        error(
            "`needs` only accepts job handles given by `@job`, got `42`, `init`, `${{ matrix.a }}`"
        )
        needs(42, "init", matrix.a)

    @job
    def other_job():
        error("no job was declared as a `needs` prerequisite of this job")
        run(j.outputs)

    @job
    def yet_another_job():
        needs(other_job)
        error("no `j` job was declared as a `needs` prerequisite of this job")
        run(j.outputs)


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
        strategy.matrix(fromJson("{}"))

    @job
    def j4():
        strategy.matrix(x=[42])
        error("`a` was not declared in the `matrix` for this job")
        step(matrix.a)

    @job
    def j5():
        error("`matrix` cannot be used in the `strategy` field defining it")
        strategy.matrix(x=[42]).max_parallel(matrix.x)
        error("`matrix` cannot be used in the `strategy` field defining it")
        strategy.matrix(y=[matrix.x])
        run("")


@expect_errors
def test_steps_errors(error):

    error("`steps` can only be used in a job, did you forget a `@job` decoration?")
    env(FOO=steps)

    # fmt: off
    @job
    # fmt: on
    def j():
        error("`steps` can only be used while constructing a step or setting outputs")
        env(FOO=steps)
        error("step `x` not defined yet in job `j`")
        run(f"echo {steps.x.outcome}")
        run("").id("x")
        error("step `y` not defined yet in job `j`")
        step("print self outcome?").run(f" {steps.y.result}").id("y")
        return steps.z.outputs
