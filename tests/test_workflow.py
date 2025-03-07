import typing

from conftest import expect
from src.pyactions.ctx import *


@expect(
    """
name: My workflow
on:
  workflow-dispatch: {}
  pull-request:
    branches:
    - main
jobs: {}
"""
)
def test_basic():
    name("My workflow")
    on.pull_request(branches=["main"])
    on.workflow_dispatch()


@expect(
    """
name: My workflow
on:
  workflow-dispatch: {}
jobs: {}
"""
)
def test_name_from_docstring():
    """My workflow"""
    on.workflow_dispatch()


@expect(
    """
on:
  pull-request:
    branches:
    - main
    paths:
    - foo/**
jobs: {}
"""
)
def test_merge():
    on.pull_request(branches=["main"])
    on.pull_request(paths=["foo/**"])


@expect(
    """
on:
  workflow-dispatch: {}
jobs:
  my_job:
    name: My job
    runs-on: ubuntu-latest
    env:
      FOO: bar
"""
)
def test_job():
    on.workflow_dispatch()

    @job
    def my_job():
        name("My job")
        env(FOO="bar")


@expect(
    """
on:
  workflow-dispatch: {}
jobs:
  my_job:
    name: My job
    runs-on: ubuntu-latest
    env:
      FOO: bar
"""
)
def test_job_name_from_docstring():
    on.workflow_dispatch()

    @job
    def my_job():
        """My job"""
        env(FOO="bar")


@expect(
    """
on:
  workflow-dispatch: {}
jobs:
  job1:
    name: First job
    runs-on: ubuntu-latest
    env:
      FOO: bar
  job2:
    name: Second job
    runs-on: ubuntu-latest
    env:
      BAZ: bazz
"""
)
def test_jobs():
    on.workflow_dispatch()

    @job
    def job1():
        name("First job")
        env(FOO="bar")

    @job
    def job2():
        name("Second job")
        env(BAZ="bazz")


@expect(
    """
on:
  workflow-dispatch: {}
jobs:
  my_job:
    runs-on: windows-latest
"""
)
def test_job_runs_on():
    on.workflow_dispatch()

    @job
    def my_job():
        runs_on("windows-latest")


@expect(
    """
on: {}
jobs:
  a_job:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        x:
        - 1
        - 2
        - 3
        y:
        - a
        - b
        - c
"""
)
def test_strategy_with_cross_matrix():
    @job
    def a_job():
        strategy.matrix(x=[1, 2, 3], y=["a", "b", "c"])


@expect(
    """
on: {}
jobs:
  a_job:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        include:
        - x: 100
          y: z
        exclude:
        - x: 1
          y: a
        x:
        - 1
        - 2
        - 3
        y:
        - a
        - b
        - c
"""
)
def test_strategy_with_include_exclude_matrix():
    @job
    def a_job():
        strategy.matrix(
            x=[1, 2, 3],
            y=["a", "b", "c"],
            exclude=[{"x": 1, "y": "a"}],
            include=[{"x": 100, "y": "z"}],
        )


@expect(
    """
on: {}
jobs:
  a_job:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        x:
        - 1
        - 2
        - 3
        y:
        - a
        - b
        - c
      fail-fast: true
      max-parallel: 5
"""
)
def test_strategy_with_fail_fast_and_max_parallel():
    @job
    def a_job():
        strategy.matrix(x=[1, 2, 3], y=["a", "b", "c"]).fail_fast().max_parallel(5)


@expect(
    """
on: {}
jobs:
  a_job:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        x:
        - 1
        - 2
        - 3
        y:
        - a
        - b
        - c
"""
)
def test_matrix_shortcut():
    @job
    def a_job():
        matrix(x=[1, 2, 3], y=["a", "b", "c"])


@expect(
    """
on:
  workflow-dispatch: {}
jobs:
  test_strategy_in_workflow:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        x:
        - 1
        - 2
        - 3
"""
)
def test_strategy_in_workflow():
    on.workflow_dispatch()
    strategy.matrix(x=[1, 2, 3])


@expect(
    """
on:
  workflow-dispatch: {}
env:
  WORKFLOW_ENV: 1
jobs:
  test_runs_on_in_workflow:
    runs-on: macos-latest
    env:
      JOB_ENV: 2
"""
)
def test_runs_on_in_workflow():
    on.workflow_dispatch()
    env(WORKFLOW_ENV=1)
    runs_on("macos-latest")
    env(JOB_ENV=2)


@expect(
    """
name: Foo bar
on:
  workflow-dispatch: {}
jobs:
  test_runs_on_in_worfklow_with_name:
    name: Foo bar
    runs-on: macos-latest
"""
)
def test_runs_on_in_worfklow_with_name():
    name("Foo bar")
    on.workflow_dispatch()
    runs_on("macos-latest")


@expect(
    """
on:
  workflow-dispatch: {}
jobs:
  my_job:
    runs-on: ubuntu-latest
    steps:
    - name: salutations
      run: echo hello
    - run: echo $WHO
      env:
        WHO: world
    - name: catastrophe
      if: failure()
      run: echo oh no
    - use: actions/checkout@v4
      with:
        ref: dev
    - use: ./my_action
      with:
        arg1: foo
        arg2: bar
    - use: ./my_other_action
      with:
        arg1: foo
        arg2: bar
"""
)
def test_steps():
    on.workflow_dispatch()

    @job
    def my_job():
        step.run("echo hello").name("salutations")
        run("echo $WHO").env(WHO="world")
        step("catastrophe").run("echo oh no").if_("failure()")
        step.use("actions/checkout@v4").with_(ref="dev")
        use("./my_action").with_(arg1="foo", arg2="bar")
        use("./my_other_action", arg1="foo", arg2="bar")


@expect(
    """
on:
  workflow-dispatch:
    inputs:
      foo:
        description: a foo
        required: true
        type: string
      bar:
        description: a bar
        required: false
        type: boolean
      baz:
        required: false
        default: b
        type: choice
        options:
        - a
        - b
        - c
      an_env:
        required: false
        type: environment
jobs: {}
"""
)
def test_workflow_dispatch_inputs():
    on.workflow_dispatch.input("foo", description="a foo", required=True).input(
        "bar", "a bar", type="boolean"
    )
    on.workflow_dispatch.input(
        "baz", type="choice", options=["a", "b", "c"], default="b"
    )
    on.workflow_dispatch.input("an_env", type="environment")


@expect(
    """
on:
  workflow-call:
    inputs:
      foo:
        required: true
        type: string
      bar:
        required: false
        type: boolean
      baz:
        required: false
        default: b
        type: choice
        options:
        - a
        - b
        - c
    secrets:
      token:
        required: true
      auth:
        description: auth if provided
        required: false
jobs: {}
"""
)
def test_workflow_call():
    (
        on.workflow_call.input("foo", required=True)
        .input("bar", type="boolean")
        .input("baz", type="choice", options=["a", "b", "c"], default="b")
        .secret("token", required=True)
        .secret("auth", "auth if provided")
    )


@expect(
    """
on:
  workflow-call:
    inputs:
      foo:
        description: a foo
        required: true
        type: string
      bar:
        required: false
        default: 42
        type: number
  workflow-dispatch:
    inputs:
      foo:
        description: a foo
        required: true
        type: string
      bar:
        required: false
        default: 42
        type: number
jobs: {}
"""
)
def test_inputs():
    input("foo", description="a foo", required=True)
    input("bar", type="number", default=42)


@expect(
    """
on:
  workflow-call:
    inputs:
      foo:
        description: a foo
        required: false
        type: string
jobs: {}
"""
)
def test_trigger_removal():
    input("foo", "a foo")
    on.workflow_dispatch(None)


@expect(
    """
on:
  workflow-call:
    inputs:
      foo:
        description: a foo
        required: false
        type: string
  workflow-dispatch:
    inputs:
      foo:
        description: a foo
        required: false
        type: string
jobs:
  test_use_input_as_expr:
    runs-on: ubuntu-latest
    steps:
    - run: foo is ${{ inputs.foo }}
"""
)
def test_use_input_as_expr():
    foo = input("foo", "a foo")
    run(f"foo is {foo}")


@expect(
    """
on:
  workflow-call:
    inputs:
      foo:
        description: a foo
        required: true
        type: number
      bar:
        required: true
        type: choice
        options:
        - apple
        - orange
        - banana
      c:
        required: true
        type: choice
        options:
        - one
        - two
      baz:
        required: false
        default: 42
        type: number
  workflow-dispatch:
    inputs:
      foo:
        description: a foo
        required: true
        type: number
      bar:
        required: true
        type: choice
        options:
        - apple
        - orange
        - banana
      c:
        required: true
        type: choice
        options:
        - one
        - two
      baz:
        required: false
        default: 42
        type: number
jobs:
  test_inputs_from_parameters:
    runs-on: ubuntu-latest
    steps:
    - run: foo is ${{ inputs.foo }}
"""
)
def test_inputs_from_parameters(foo: Input[int], bar, c: Choice["one", "two"], baz=42):
    foo.description = "a foo"
    bar.type = "choice"
    bar.options = ["apple", "orange", "banana"]
    run(f"foo is {foo}")


@expect(
    """
on: {}
jobs:
  test_id:
    runs-on: ubuntu-latest
    steps:
    - id: one
      run: one
    - id: y-1
      run: ${{ steps.one.outputs.foo }}
    - id: y
      run: ${{ steps.y-1.outcome }}
    - run: ${{ steps.y.result }}
"""
)
def test_id():
    x = step.id("one").run("one")
    y = step.run(x.outputs.foo)
    z = step.id("y").run(y.outcome)
    step.run(z.result)
