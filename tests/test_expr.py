import pytest

from src.pyactions.ctx import GenerationError
from src.pyactions.expr import *


def test_ops():
    a = Expr("a")
    b = Expr("b")
    c = Expr("c")

    assert str(a | (b & c)) == "${{ a || b && c }}"
    assert str((a | b) & c) == "${{ (a || b) && c }}"
    assert str(~a | b) == "${{ !a || b }}"
    assert str(~(a | b)) == "${{ !(a || b) }}"
    assert str((a & b)[0]) == "${{ (a && b)[0] }}"
    assert str((a & b).x) == "${{ (a && b).x }}"
    assert str((a == b) | (a < c)) == "${{ a == b || a < c }}"
    assert str((0 != a) & (a >= c) & (b > c)) == "${{ a != 0 && a >= c && b > c }}"


def test_no_bool():
    a = Expr("a")

    with pytest.raises(GenerationError):
        a and a

    with pytest.raises(GenerationError):
        a or a
