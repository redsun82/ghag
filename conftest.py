import dataclasses
import typing

import pytest

from src.pyactions.ctx import workflow
from src.pyactions import generate
import pathlib
import inspect


def pytest_addoption(parser):
    parser.addoption("--learn", action="store_true")


@dataclasses.dataclass(frozen=True)
class _Call:
    name: str
    file: pathlib.Path
    startline: int
    endline: int

    @classmethod
    def get(cls):
        frame = inspect.getframeinfo(inspect.currentframe().f_back)
        call_frame = inspect.getframeinfo(inspect.currentframe().f_back.f_back)
        return cls(
            frame.function,
            pathlib.Path(call_frame.filename),
            call_frame.positions.lineno,
            call_frame.positions.end_lineno,
        )

    def __str__(self):
        return f"{self.name}@{self.file}:{self.startline}:{self.endline}"


_learn = pytest.StashKey[list[tuple[_Call, str]]]()


def pytest_configure(config: pytest.Config):
    config.stash[_learn] = []


def expect(expected: str | None = None):
    assert isinstance(expected, str | None), "replace @expect with @expect()"
    expected = expected and expected.lstrip("\n")
    call = _Call.get()

    def decorator(f):
        def wrapper(pytestconfig: pytest.Config):
            wf = workflow(f)
            output = generate(wf, pathlib.Path(inspect.getfile(f)).parent)
            with open(output) as out:
                actual = [l.rstrip("\n") for l in out]
            if expected is None or pytestconfig.getoption("--learn"):
                pytestconfig.stash[_learn].append((call, "\n".join(actual)))
            else:
                assert actual == expected.splitlines()
            output.unlink()

        return wrapper

    return decorator


def pytest_unconfigure(config):
    changes = {}
    for call, expected in config.stash[_learn]:
        changes.setdefault(call.file, []).append(
            (call.startline, call.endline, expected)
        )
    for v in changes.values():
        v.sort()
    for f, v in changes.items():
        bkp = f.with_suffix(f"{f.suffix}.bkp")
        f.rename(bkp)
        with open(bkp) as input, open(f, "w") as output:
            input = iter(input)
            current = 1
            for startline, endline, expected in v:
                for _ in range(startline - current):
                    output.write(next(input))
                for _ in range(endline - startline + 1):
                    next(input)
                print(f'@expect(\n    """\n{expected}\n"""\n)', file=output)
                current = endline + 1
            for line in input:
                output.write(line)
