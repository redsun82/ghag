import pytest

from src.pyactions.ctx import GenerationError
from src.pyactions.expr import *
from src.pyactions.expr import expr_function


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

    x._activate("a")
    assert str(x.a) == "${{ x.a }}"

    x._activate("y")
    assert str(x.y) == "${{ x.y }}"

    with pytest.raises(ValueError):
        _ = x.y.b

    x.y._activate("b")
    assert str(x.y.b) == "${{ x.y.b }}"

    x._clear()
    with pytest.raises(ValueError):
        _ = x.a

    with pytest.raises(ValueError):
        _ = x.y

    x._activate("y")
    with pytest.raises(ValueError):
        _ = x.y.b


def test_map_context():
    class X(Context):
        a = Field(MapContext)

        class B(Context):
            c = Field()

        b = Field(MapContext, B)

    x = X("x")

    assert not x.a._has("foo")
    with pytest.raises(ValueError):
        _ = x.a.foo

    assert not x.b._has("bar")
    with pytest.raises(ValueError):
        _ = x.b.bar

    x.a._activate("foo")
    x.b._activate("bar")

    assert x.a._has("foo")
    assert str(x.a.foo) == "${{ x.a.foo }}"
    assert x.b._has("bar")
    assert str(x.b.bar) == "${{ x.b.bar }}"
    assert str(x.b.bar.c) == "${{ x.b.bar.c }}"

    x._clear()

    assert not x.a._has("foo")
    with pytest.raises(ValueError):
        _ = x.a.foo

    assert not x.b._has("bar")
    with pytest.raises(ValueError):
        _ = x.b.bar


def test_free_map_context():
    x = MapContext("x")
    x._activate_all()
    assert x._has("foo")
    assert str(x.foo) == "${{ x.foo }}"

    x._clear()

    assert not x._has("foo")
    with pytest.raises(ValueError):
        _ = x.foo


def test_functions():
    f = expr_function("foo", 3)
    x = Expr("x")
    assert str(f(1, x, "s's")) == "${{ foo(1, x, 's''s') }}"
    for trial in (lambda: f(1), lambda: f(1, 2, 3, 4), lambda: f(x=1, y=2, z=3)):
        with pytest.raises(ValueError):
            trial()
