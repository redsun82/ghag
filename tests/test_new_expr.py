import pytest

from src.pyactions.new.expr import *
import unittest.mock


def test_ops():
    a = RefExpr("a")
    b = RefExpr("b")
    c = RefExpr("c")

    assert instantiate(a | (b & c)) == "${{ a || b && c }}"
    assert instantiate((a | b) & c) == "${{ (a || b) && c }}"
    assert instantiate(~a | b) == "${{ !a || b }}"
    assert instantiate(~(a | b)) == "${{ !(a || b) }}"
    assert instantiate((a & b)[0]) == "${{ (a && b)[0] }}"
    assert instantiate((a & b).x) == "${{ (a && b).x }}"
    assert instantiate((a == b) | (a < c)) == "${{ a == b || a < c }}"
    assert (
        instantiate((0 != a) & (a >= c) & (b > c)) == "${{ a != 0 && a >= c && b > c }}"
    )


def test_refs():
    a = RefExpr("a")
    b = RefExpr("b")
    c = RefExpr("c")

    assert refs(a) == {a}
    assert refs(a | (b & c)) == {a, b, c}
    assert refs(~(a | b)) == {a, b}
    assert refs(a[b][0]) == {a, b}


def test_ref_dot():
    a = RefExpr("a")
    dot = a.x.y
    assert dot is RefExpr("a", "x", "y")
    assert instantiate(dot) == "${{ a.x.y }}"


# def test_no_bool():
#     a = Expr("a")
#
#     with pytest.raises(ValueError):
#         a and a
#
#     with pytest.raises(ValueError):
#         a or a
#
#
# def test_simple_context():
#     class X(Context):
#         a = Expr()
#         b = Expr()
#
#     x = X("x")
#
#     assert str(x) == "${{ x }}"
#     assert str(x.a) == "${{ x.a }}"
#     assert str(x.b) == "${{ x.b }}"
#
#     with pytest.raises(ValueError):
#         _ = x.c
#
#
# def test_nested_context():
#     class X(Context):
#         class Y(Context):
#             b = Expr()
#             a = Expr()
#
#         y = Y()
#
#     x = X("x")
#
#     assert str(x.y) == "${{ x.y }}"
#     assert str(x.y.a) == "${{ x.y.a }}"
#     assert str(x.y.b) == "${{ x.y.b }}"
#
#     with pytest.raises(ValueError):
#         _ = x.c
#
#
# def test_potential_fields():
#     class X(Context):
#         a = Inactive()
#
#         class Y(Context):
#             b = Inactive()
#
#         y = Inactive(Y)
#
#         class Z(Context):
#             c = Inactive()
#
#         z = Z()
#
#     x = X("x")
#
#     with pytest.raises(ValueError):
#         _ = x.a
#
#     with pytest.raises(ValueError):
#         _ = x.y
#
#     x._activate("a")
#     assert str(x.a) == "${{ x.a }}"
#
#     x._activate("y")
#     assert str(x.y) == "${{ x.y }}"
#
#     with pytest.raises(ValueError):
#         _ = x.y.b
#
#     x.y._activate("b")
#     assert str(x.y.b) == "${{ x.y.b }}"
#
#     x._clear()
#     with pytest.raises(ValueError):
#         _ = x.a
#
#     with pytest.raises(ValueError):
#         _ = x.y
#
#     x._activate("y")
#     with pytest.raises(ValueError):
#         _ = x.y.b
#
#
# def test_map_context():
#     class X(Context):
#         a = MapContext()
#
#         class B(Context):
#             c = Expr()
#
#         b = MapContext(fieldcls=B)
#
#     x = X("x")
#
#     assert not x.a._has("foo")
#     with pytest.raises(ValueError):
#         _ = x.a.foo
#
#     assert not x.b._has("bar")
#     with pytest.raises(ValueError):
#         _ = x.b.bar
#
#     x.a._activate("foo")
#     x.b._activate("bar")
#
#     assert x.a._has("foo")
#     assert str(x.a.foo) == "${{ x.a.foo }}"
#     assert x.b._has("bar")
#     assert str(x.b.bar) == "${{ x.b.bar }}"
#     assert str(x.b.bar.c) == "${{ x.b.bar.c }}"
#
#     x._clear()
#
#     assert not x.a._has("foo")
#     with pytest.raises(ValueError):
#         _ = x.a.foo
#
#     assert not x.b._has("bar")
#     with pytest.raises(ValueError):
#         _ = x.b.bar
#
#
# def test_free_map_context():
#     x = MapContext(value="x")
#     x._activate_all()
#     assert x._has("foo")
#     assert str(x.foo) == "${{ x.foo }}"
#
#     x._clear()
#
#     assert not x._has("foo")
#     with pytest.raises(ValueError):
#         _ = x.foo
#
#
def test_functions():
    f = function("foo", 3)
    x = RefExpr("x")
    assert instantiate(f(1, x, "s's")) == "${{ foo(1, x, 's''s') }}"
    assert (
        instantiate(f(1) & 1)
        == "${{ error('wrong number of arguments to `foo`, expected 3, got 1') }}"
    )
    assert (
        instantiate(f(1, 2, 3, 4) == 1)
        == "${{ error('wrong number of arguments to `foo`, expected 3, got 4') }}"
    )
    assert (
        instantiate(f(x=1, y=2, z=3)[0])
        == "${{ error('unexpected keyword arguments to `foo`, expected 3 positional arguments') }}"
    )

    assert refs(f(1, x, "s's")) == {x}


#
# def test_error_expr():
#     e = unittest.mock.Mock()
#     with on_error(e):
#         ee = Expr(_error="an error")
#         e.assert_not_called()
#         _ = str(ee)
#         e.assert_called_once_with("an error")
#         e.reset_mock()
#         _ = str(ee.x.y[0] & 3 | True == "x")
#         e.assert_not_called()
