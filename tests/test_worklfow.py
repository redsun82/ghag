from conftest import expect
from src.pyactions.ctx import *


@expect(
    """
on:
  pull-request:
    branches:
    - main
  workflow-dispatch: {}
jobs: {}
"""
)
def test_basic():
    on.pull_request(branches=["main"])
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
