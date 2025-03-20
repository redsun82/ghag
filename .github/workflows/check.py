from gag.ctx import *


@workflow
def check():
    on.pull_request().push()
    step("Checkout").uses("actions/checkout@v4")
    step("Check formatting").run(
        "uv run black --check ."
    ).continue_on_error().ensure_id()
    step("Run tests").run("uv run pytest").continue_on_error().ensure_id()
    step("Fail").if_(contains(steps._.outcome, "failed")).run("exit 1")
