from src.pyactions.workflow import *
from src.pyactions.element import *


def test_default():
    w = Workflow()
    assert asobj(w) == {"jobs": {}, "on": {}}
