import functools

from src.pyactions.ctx import workflow
from src.pyactions import generate
import difflib
import pathlib
import inspect

def pytest_addoption(parser):
    parser.addoption(
        "--learn", action="store_true", help="learn test output"
    )

def check_generation(f, learn=False):
    wf = workflow(f)
    output = generate(wf, pathlib.Path(inspect.getfile(f)).parent)
    expected_file = output.with_suffix(".expected.yml")
    if learn:
        output.rename(expected_file)
        return
    with open(output) as out:
        actual = [l.rstrip("\n") for l in out]
    assert expected_file.exists(), f"{expected_file.name} not found"
    with open(expected_file) as expected:
        expected = [l.rstrip("\n") for l in expected]
    assert actual == expected, output.name
    output.unlink()

def generation_test(f):
    def wrapper(pytestconfig):
        check_generation(f, pytestconfig.getoption("--learn"))
    return wrapper