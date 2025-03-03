from conftest import generation_test
from src.pyactions.ctx import *

@generation_test
def test_basic():
    on.pull_request(branches=["main"])
    on.workflow_dispatch()

@generation_test
def test_merge():
    on.pull_request(branches=["main"])
    on.pull_request(paths=["foo/**"])
