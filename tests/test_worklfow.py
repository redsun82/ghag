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
