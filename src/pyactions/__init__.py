import argparse
import logging
import typing
import pathlib
from ruamel.yaml import YAML

from .ctx import WorkflowInfo

def generate(w: WorkflowInfo, dir: pathlib.Path) -> pathlib.Path:
    output = (dir / w.id).with_suffix(".yml")
    w = w.instantiate().asdict()
    yaml = YAML()
    with open(output, "w") as out:
        yaml.dump(w, out)
    return output

def options(args: typing.Sequence[str] = None):
    p = argparse.ArgumentParser(description="Generate Github Actions workflows")
    p.add_argument("--output-directory", "-D", type=pathlib.Path)
    p.add_argument(dest="inputs", nargs="*")
    return p.parse_args(args)


def main(args: typing.Sequence[str] = None) -> None:
    opts = options(args)
    logging.info(opts.__dict__)
