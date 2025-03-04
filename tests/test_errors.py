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
