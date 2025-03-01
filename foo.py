from ctx import *

# step.run.set_defaults(shell="bash")
#
# @step.use.set_defaults
# def defaults(u):
#     if u.startswith("actions/upload@"):
#         return {"with_": {"retention_days": 5}}
#     if u.startswith("actions/checkout@"):
#         return {"name": "Checkout"}
#     return None
#
# job.set_defaults(timeout_minutes=15)


@workflow
def my_workflow():
    on.pull_request(branches=["main"], paths=["**/bla"])
    on.workflow_dispatch()

    @job
    def do_something():
        matrix(runner=["ubuntu-latest", "windows-latest"])
        # runs_on(matrix.runner)
        runs_on("ubuntu-latest")
        step.use("actions/checkout@v4")
        step("hello").run(
            """
            echo hello
            if true; then
                echo world
            fi
        """
        )
        step.if_(~(failed() | skipped() & failed() | "foo's")).name("Oh no").run(
            "echo catastrophe!"
        )

        # output_step = step("Get output").output(my_output=42)
        # step("Use output").run(f"Output was {output_step.outputs.my_output}")
        # with step.set_defaults(if_=failed()):
        #     step("Oh no").run("echo catastrophe!")
        #     step("Upload").use("actions/upload@v4").with_(
        #         name="debug",
        #         path="**/*.log",
        #     )
