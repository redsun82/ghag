import argparse
import importlib.util
import logging
import sys
import typing
import pathlib
import difflib

from ruamel.yaml import YAML, CommentedMap

from .ctx import WorkflowInfo, GenerationError
import functools
import colorlog

yaml = YAML()
yaml.default_flow_style = None


class DiffError(Exception):
    def __init__(self, diff):
        super().__init__("generated workflow does not match expected")
        self.errors = diff


def generate_workflow(
    w: WorkflowInfo, dir: pathlib.Path, check=False
) -> pathlib.Path | None:
    input = f"{w.file.name}::{w.spec.__name__}"
    output = (dir / w.id).with_suffix(".yml")
    tmp = output.with_suffix(".yml.tmp")
    w = w.instantiate().asdict()
    w = CommentedMap(w)
    w.yaml_set_start_comment(f"generated from {input}")
    with open(tmp, "w") as out:
        yaml.dump(w, out)
    if check:
        if output.exists():
            with open(output) as current:
                current = [*current]
        else:
            current = []
        with open(tmp) as new:
            new = [*new]
        diff = list(difflib.unified_diff(current, new, str(output), str(tmp)))
        if diff:
            raise DiffError([l.rstrip("\n") for l in diff])
        tmp.unlink()
    else:
        tmp.rename(output)
    return output


@functools.cache
def discover_workflows_dir() -> pathlib.Path:
    def iter():
        cwd = pathlib.Path.cwd()
        yield cwd
        yield from cwd.parents

    for dir in iter():
        if dir.joinpath(".github").exists() or dir.joinpath(".git").exists():
            return relativized_path(dir.joinpath(".github", "workflows"))
    raise FileNotFoundError(
        "no `.github` or `.git` directory found in any ancestor of the current directory"
    )


def relativized_path(p: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(p)
    if not p.is_absolute():
        return p
    cwd = pathlib.Path.cwd()
    try:
        return p.relative_to(cwd)
    except ValueError:
        return p


def generate(opts: argparse.Namespace):
    sys.path.extend(map(str, opts.includes))
    sys.modules["gag"] = sys.modules[__name__]
    inputs = opts.inputs or opts.includes
    failed = False
    found = False
    for i in inputs:
        logging.debug(f"@ {i}")
        for f in i.glob("*.py"):
            logging.debug(f"← {f}")
            spec = importlib.util.spec_from_file_location(f.name, str(f))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for k, v in mod.__dict__.items():
                if isinstance(v, WorkflowInfo):
                    found = True
                    try:
                        output = generate_workflow(
                            v, opts.output_directory, check=opts.check
                        )
                        logging.info(f"{'✅' if opts.check else '→'} {output}")
                    except (GenerationError, DiffError) as e:
                        failed = True
                        for error in e.errors:
                            logging.error(error)
    if not found:
        logging.error("no workflows found")
        return 2
    if failed:
        return 1
    return 0


def options(args: typing.Sequence[str] = None):
    p = argparse.ArgumentParser(description="Generate Github Actions workflows")

    def common_opts(parser):
        parser.add_argument(
            "--output-directory",
            "-D",
            type=relativized_path,
            metavar="DIR",
            help="Where output files should be written (`.github/workflows` by default)",
        )
        parser.add_argument(
            "--include",
            "-I",
            type=relativized_path,
            metavar="DIR",
            action="append",
            dest="includes",
            help="Add DIR to the system include paths. Can be repeated. If none are provided `.github/workflows` is used. Includes are also used as default inputs.",
        )
        parser.add_argument("--verbose", "-v", action="store_true")
        parser.add_argument(
            "inputs",
            nargs="*",
            help="input directories to process",
            type=relativized_path,
        )
        parser.add_argument("--check", "-C", action="store_true")

    common_opts(p)
    p.set_defaults(command=generate)
    commands = p.add_subparsers()
    gen = commands.add_parser(
        "generate", aliases=["g", "gen"], help="generate workflows"
    )
    gen.set_defaults(command=generate)
    common_opts(gen)
    ret = p.parse_args(args)
    if not ret.command:
        p.print_help()
        sys.exit(0)
    ret.output_directory = ret.output_directory or discover_workflows_dir()
    ret.includes = ret.includes or [discover_workflows_dir()]
    return ret


def main(args: typing.Sequence[str] = None) -> int:
    opts = options(args)
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "{log_color}{levelname: <8}{reset} {message_log_color}{message}",
            secondary_log_colors={
                "message": {
                    "DEBUG": "white",
                    "WARNING": "bold",
                    "ERROR": "bold",
                    "CRITICAL": "bold",
                },
            },
            style="{",
        )
    )
    logging.basicConfig(
        level=logging.INFO if not opts.verbose else logging.DEBUG, handlers=[handler]
    )
    logging.debug(opts.__dict__)
    return opts.command(opts)
