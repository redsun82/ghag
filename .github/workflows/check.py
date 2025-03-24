from ghgen.ctx import *


@workflow
def check():
    on.pull_request().push()
    use("actions/checkout@v4")
    use("astral-sh/setup-uv@v5")
    step("Check formatting").run("uv run black --check .")
    step("Run tests").if_(~cancelled()).run("uv run pytest")
    step("Check generation").if_(~cancelled()).run("uv run gh-gen --check")
