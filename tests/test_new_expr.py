import pytest

from pyactions.expr import MapContext
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


def test_ref_dot():
    a = RefExpr("a")
    dot = a.x.y
    assert dot is RefExpr("a", "x", "y")
    assert instantiate(dot) == "${{ a.x.y }}"


def test_refs():
    a = RefExpr("a")
    b = RefExpr("b")
    c = RefExpr("c")

    assert refs(a) == {a}
    assert refs(a | (b & c)) == {a, b, c}
    assert refs(~(a | b)) == {a, b}
    assert refs(a[b][0]) == {a, b}


def test_refs_on_strings():
    a = RefExpr("a")
    b = RefExpr("b")
    c = RefExpr("c")

    assert refs(f"-- {a} __ {b} || {c} >>") == {a, b, c}
    assert refs(
        {
            "FOO": f"/{a}",
            "BAR": f"/{a}/{b}",
            "BAZ": c,
        }
    ) == {a, b, c}
    assert refs([f"<{a}>", f"<{b}, {a}>", c]) == {a, b, c}


def test_no_bool():
    error_handler = unittest.mock.Mock()
    with on_error(error_handler):
        a = RefExpr("a")
        assert a
        error_handler.assert_called_once()


def test_simple_context():
    x = Context(
        "x",
        structure={
            "a": None,
            "b": None,
        },
    )

    assert instantiate(x) == "${{ x }}"
    assert instantiate(x.a) == "${{ x.a }}"
    assert instantiate(x.b) == "${{ x.b }}"

    error_handler = unittest.mock.Mock()
    with on_error(error_handler):
        _ = x.c
        _ = x.b.whatever
        error_handler.assert_has_calls(
            [
                unittest.mock.call("`c` not available in `x`"),
                unittest.mock.call("`whatever` not available in `x.b`"),
            ]
        )


def test_nested_context():
    x = Context(
        "x",
        structure={
            "a": None,
            "b": None,
            "y": {
                "c": None,
                "d": None,
            },
        },
    )

    assert instantiate(x.a) == "${{ x.a }}"
    assert instantiate(x.b) == "${{ x.b }}"
    assert instantiate(x.y.c) == "${{ x.y.c }}"
    assert instantiate(x.y.d) == "${{ x.y.d }}"

    error_handler = unittest.mock.Mock()
    with on_error(error_handler):
        _ = x.c
        _ = x.y.bla
        _ = x.y.a
        error_handler.assert_has_calls(
            [
                unittest.mock.call("`c` not available in `x`"),
                unittest.mock.call("`bla` not available in `x.y`"),
                unittest.mock.call("`a` not available in `x.y`"),
            ]
        )


def test_map_context():
    x = Context(
        "x",
        structure={
            "a": None,
            "b": {
                "*": None,
            },
            "*": {
                "foo": None,
                "bar": {
                    "baz": None,
                },
            },
        },
    )

    assert instantiate(x) == "${{ x }}"
    assert instantiate(x.a) == "${{ x.a }}"
    assert instantiate(x.b) == "${{ x.b }}"
    assert instantiate(x.b.whatever) == "${{ x.b.whatever }}"
    assert instantiate(x.baz) == "${{ x.baz }}"
    assert instantiate(x.baz.foo) == "${{ x.baz.foo }}"
    assert instantiate(x.baz.bar) == "${{ x.baz.bar }}"
    assert instantiate(x.baz.bar.baz) == "${{ x.baz.bar.baz }}"

    error_handler = unittest.mock.Mock()
    with on_error(error_handler):
        _ = x.b.foo.whatever
        _ = x.baz.foo.whatever
        _ = x.baz.bar.whatever
        error_handler.assert_has_calls(
            [
                unittest.mock.call("`whatever` not available in `x.b.foo`"),
                unittest.mock.call("`whatever` not available in `x.baz.foo`"),
                unittest.mock.call("`whatever` not available in `x.baz.bar`"),
            ]
        )


def test_functions():
    f = function("foo", 3)
    x = RefExpr("x")
    assert instantiate(f(1, x, "s's")) == "${{ foo(1, x, 's''s') }}"
    for wrong, expected_error in (
        (lambda: f(1), "wrong number of arguments to `foo`, expected 3, got 1"),
        (
            lambda: f(1, 2, 3, 4),
            "wrong number of arguments to `foo`, expected 3, got 4",
        ),
        (
            lambda: f(x=1, y=2, z=3),
            "unexpected keyword arguments to `foo`, expected 3 positional arguments",
        ),
    ):
        error_handler = unittest.mock.Mock()
        with on_error(error_handler):
            call = wrong()
            error_handler.assert_called_once_with(expected_error)
            assert instantiate(call) == f"${{{{ error('{expected_error}') }}}}"

    assert refs(f(1, x, "s's")) == {x}


def test_error_expr():
    error_handler = unittest.mock.Mock()
    with on_error(error_handler):
        e = ErrorExpr("an error")
        error_handler.assert_not_called()
        assert instantiate(e) == "${{ error('an error') }}"
        error_handler.assert_called_once_with("an error")
        error_handler.reset_mock()
        assert instantiate(e.x.y[0] & 3 | True == "x") == "${{ error('an error') }}"
        error_handler.assert_not_called()


def test_default_error():
    with pytest.raises(ValueError) as e:
        _ = ~ErrorExpr("an error")
    assert e.value.args == ("an error",)
