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

    with pytest.raises(ValueError):
        a and a

    with pytest.raises(ValueError):
        a or a


def test_simple_context():
    class X(Context):
        a = Field()
        b = Field()

    x = X("x")

    assert str(x) == "${{ x }}"
    assert str(x.a) == "${{ x.a }}"
    assert str(x.b) == "${{ x.b }}"

    with pytest.raises(ValueError):
        _ = x.c


def test_nested_context():
    class X(Context):
        class Y(Context):
            a = Field()
            b = Field()

        y = Field(Y)

    x = X("x")

    assert str(x.y) == "${{ x.y }}"
    assert str(x.y.a) == "${{ x.y.a }}"
    assert str(x.y.b) == "${{ x.y.b }}"

    with pytest.raises(ValueError):
        _ = x.c


def test_potential_fields():
    class X(Context):
        a = PotentialField()

        class Y(Context):
            b = PotentialField()

        y = PotentialField(Y)

        class Z(Context):
            c = PotentialField()

        z = Field(Z)

    x = X("x")

    with pytest.raises(ValueError):
        _ = x.a

    with pytest.raises(ValueError):
        _ = x.y

    x.activate("a")
    assert str(x.a) == "${{ x.a }}"

    x.activate("y")
    assert str(x.y) == "${{ x.y }}"

    with pytest.raises(ValueError):
        _ = x.y.b

    x.y.activate("b")
    assert str(x.y.b) == "${{ x.y.b }}"

    x.clear()
    with pytest.raises(ValueError):
        _ = x.a

    with pytest.raises(ValueError):
        _ = x.y

    x.activate("y")
    with pytest.raises(ValueError):
        _ = x.y.b


def test_map_context():
    class X(Context):
        a = Field(MapContext)

        class B(Context):
            c = Field()

        b = Field(MapContext, B)

    x = X("x")

    with pytest.raises(ValueError):
        _ = x.a.foo

    with pytest.raises(ValueError):
        _ = x.b.bar

    x.a.activate("foo")
    x.b.activate("bar")

    assert str(x.a.foo) == "${{ x.a.foo }}"
    assert str(x.b.bar) == "${{ x.b.bar }}"
    assert str(x.b.bar.c) == "${{ x.b.bar.c }}"

    x.clear()

    with pytest.raises(ValueError):
        _ = x.a.foo

    with pytest.raises(ValueError):
        _ = x.b.bar
