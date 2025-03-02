from conftest import generation_test
from src.pyactions.ctx import *

@generation_test
def test_basic():
    on.pull_request(branches=["main"])
    on.workflow_dispatch()
