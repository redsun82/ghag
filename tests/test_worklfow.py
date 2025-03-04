from conftest import expect
from src.pyactions.ctx import *


@expect(
    """
name: My workflow
on:
  pull-request:
    branches:
    - main
  workflow-dispatch: {}
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
on:
  workflow-dispatch: {}
jobs:
  with_cross_matrix:
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
  with_include_exclude_matrix:
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
  with_fail_fast_and_max_parallel:
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
def test_strategy():
    on.workflow_dispatch()

    @job
    def with_cross_matrix():
        strategy.matrix(x=[1, 2, 3], y=["a", "b", "c"])

    @job
    def with_include_exclude_matrix():
        strategy.matrix(
            x=[1, 2, 3],
            y=["a", "b", "c"],
            exclude=[{"x": 1, "y": "a"}],
            include=[{"x": 100, "y": "z"}],
        )

    @job
    def with_fail_fast_and_max_parallel():
        strategy.matrix(x=[1, 2, 3], y=["a", "b", "c"]).fail_fast().max_parallel(5)


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
