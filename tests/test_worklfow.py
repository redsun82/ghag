from src.pyactions.workflow import *
from src.pyactions.element import *
from src.pyactions.ctx import *
from src.pyactions import generate

import subprocess


def test_default(request):
    @workflow
    def test_default():
        env(
            FOO="bar",
        )
        on.pull_request(branches=["main"])
        on.workflow_dispatch()
        env(
            BAZ="bazz",
        )
    output = generate(test_default, request.path.parent)
    subprocess.run(["diff", "-u", output, output.with_suffix(".expected.yml")],
                   check=True)