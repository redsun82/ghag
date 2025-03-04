import dataclasses

import pytest

from src.pyactions.ctx import workflow, GenerationError
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
    assert not callable(expected), "replace @expect with @expect()"
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

def expect_errors(expected: str | None = None):
    assert not callable(expected), "replace @expect_errors with @expect_errors()"
    expected = expected and expected.lstrip("\n")
    call = _Call.get()

    def decorator(f):
        def wrapper(request: pytest.FixtureRequest):
            wf = workflow(f)
            with pytest.raises(GenerationError) as e:
                generate(wf, pathlib.Path(request.node.path.parent))
            for err in e.value.errors:
                err.filename = str(pathlib.Path(err.filename).relative_to(request.node.path.parent))
            actual = map(str, e.value.errors)
            if expected is None or request.config.getoption("--learn"):
                request.config.stash[_learn].append((call, "\n".join(actual)))
            else:
                assert actual == expected.splitlines()

        return wrapper

    return decorator


def pytest_unconfigure(config):
    changes = {}
    for call, expected in config.stash[_learn]:
        changes.setdefault(call.file, []).append(
            (call.startline, call.endline, call.name, expected)
        )
    for v in changes.values():
        v.sort()
    for f, v in changes.items():
        bkp = f.with_suffix(f"{f.suffix}.bkp")
        f.rename(bkp)
        with open(bkp) as input, open(f, "w") as output:
            input = iter(input)
            current = 1
            for startline, endline, name, expected in v:
                for _ in range(startline - current):
                    output.write(next(input))
                for _ in range(endline - startline + 1):
                    next(input)
                print(f'@{name}(\n    """\n{expected}\n"""\n)', file=output)
                current = endline + 1
            for line in input:
                output.write(line)
