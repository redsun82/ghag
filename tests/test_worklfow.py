from src.pyactions.workflow import *
from src.pyactions.element import *
from src.pyactions.ctx import *


def test_default():
    @workflow
    def wf():
        env(
            FOO="bar",
        )
        on.pull_request(branches=["main"])
        on.workflow_dispatch()
        env(
            BAZ="bazz",
        )

    w = wf.instantiate()
    assert asobj(w) == {
        "jobs": {},
        "on": {"pull-request": {"branches": ["main"]}, "workflow-dispatch": {}},
        "env": {"FOO": "bar", "BAZ": "bazz"},
    }
