from ghag.ctx import *


@workflow
def check():
    on.pull_request().push()
    use("actions/checkout@v4")
    use("astral-sh/setup-uv@v5")
    step("Check formatting").run("uv run black --check .")
    step("Run tests").run("uv run pytest").if_(~cancelled())
    step("Check generation").run("uv run ghag --check").if_(~cancelled())
