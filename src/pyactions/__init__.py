import argparse
import logging
import typing


def options(args: typing.Sequence[str]=None):
    p = argparse.ArgumentParser(description="Generate Github Actions workflows")
    p.add_argument("--output-directory", "-D")
    p.add_argument(dest="inputs", nargs="*")
    return p.parse_args(args)

def main(args: typing.Sequence[str]=None) -> None:
    opts = options(args)
    logging.info(opts.__dict__)
