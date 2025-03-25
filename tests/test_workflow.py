import typing

from conftest import expect
from src.ghgen.ctx import *


@expect(
    """
# generated from test_workflow.py::test_basic
name: My workflow
on:
  pull_request:
    branches:
    - main
  workflow_dispatch: {}
jobs: {}
"""
)
def test_basic():
    name("My workflow")
    on.pull_request(branches=["main"])
    on.workflow_dispatch()


@expect(
    """
# generated from test_workflow.py::test_pull_request
on:
  pull_request:
    types:
    - opened
    - reopened
    branches:
    - main
    - dev/*
    ignore-branches:
    - dev/ignore
    paths:
    - foo/**
    ignore-paths:
    - foo/bar/**
jobs: {}
"""
)
def test_pull_request():
    on.pull_request(
        types=["opened", "reopened"],
        branches=["main", "dev/*"],
        ignore_branches=["dev/ignore"],
        paths=["foo/**"],
        ignore_paths=["foo/bar/**"],
    )


@expect(
    """
# generated from test_workflow.py::test_merge
on:
  pull_request:
    branches:
    - main
    paths:
    - foo/**
jobs: {}
"""
)
def test_merge():
    on.pull_request(branches=["main"])
    on.pull_request(paths=["foo/**"])


@expect(
    """
# generated from test_workflow.py::test_job
on:
  workflow_dispatch: {}
jobs:
  my_job:
    name: My job
    env:
      FOO: bar
"""
)
def test_job():
    on.workflow_dispatch()

    @job
    def my_job():
        name("My job")
        env(FOO="bar")


@expect(
    """
# generated from test_workflow.py::test_jobs
on:
  workflow_dispatch: {}
jobs:
  job1:
    name: First job
    env:
      FOO: bar
  job2:
    name: Second job
    env:
      BAZ: bazz
"""
)
def test_jobs():
    on.workflow_dispatch()

    @job
    def job1():
        name("First job")
        env(FOO="bar")

    @job
    def job2():
        name("Second job")
        env(BAZ="bazz")


@expect(
    """
# generated from test_workflow.py::test_job_runs_on
on:
  workflow_dispatch: {}
jobs:
  my_job:
    runs-on: windows-latest
"""
)
def test_job_runs_on():
    on.workflow_dispatch()

    @job
    def my_job():
        runs_on("windows-latest")


@expect(
    """
# generated from test_workflow.py::test_multiline_run_code_dedented
on: {}
jobs:
  test_multiline_run_code_dedented:
    runs-on: ubuntu-latest
    steps:
    - run: |
        echo one
        echo two
        echo three
"""
)
def test_multiline_run_code_dedented():
    run("""
        echo one
        echo two
        echo three
    """) # fmt: skip


@expect(
    """
# generated from test_workflow.py::test_strategy_with_cross_matrix
on: {}
jobs:
  a_job:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        x: [1, 2, 3]
        y: [a, b, c]
    steps:
    - run: ${{ matrix.x }}, ${{ matrix.y }}
"""
)
def test_strategy_with_cross_matrix():
    @job
    def a_job():
        strategy.matrix(x=[1, 2, 3], y=["a", "b", "c"])
        run(f"{matrix.x}, {matrix.y}")


@expect(
    """
# generated from test_workflow.py::test_strategy_with_include_exclude_matrix
on: {}
jobs:
  a_job:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        include:
        - x: 100
          y: z
          z: 42
        exclude:
        - x: 1
          y: a
        x: [1, 2, 3]
        y: [a, b, c]
    steps:
    - run: ${{ matrix.x }}, ${{ matrix.y }}, ${{ matrix.z }}
"""
)
def test_strategy_with_include_exclude_matrix():
    @job
    def a_job():
        strategy.matrix(
            x=[1, 2, 3],
            y=["a", "b", "c"],
            exclude=[{"x": 1, "y": "a"}],
            include=[{"x": 100, "y": "z", "z": 42}],
        )
        run(f"{matrix.x}, {matrix.y}, {matrix.z}")


@expect(
    """
# generated from test_workflow.py::test_strategy_with_fail_fast_and_max_parallel
on: {}
jobs:
  a_job:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        x: [1, 2, 3]
        y: [a, b, c]
      fail-fast: true
      max-parallel: 5
    steps:
    - run: ${{ matrix.x }}, ${{ matrix.y }}
"""
)
def test_strategy_with_fail_fast_and_max_parallel():
    @job
    def a_job():
        strategy.matrix(x=[1, 2, 3], y=["a", "b", "c"]).fail_fast().max_parallel(5)
        run(f"{matrix.x}, {matrix.y}")


@expect(
    """
# generated from test_workflow.py::test_strategy_in_workflow
on:
  workflow_dispatch: {}
jobs:
  test_strategy_in_workflow:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        include:
          z: 42
        x: [1, 2, 3]
        y: [a, b]
    steps:
    - run: ${{ matrix.x }}, ${{ matrix.y }}, ${{ matrix.z }}
"""
)
def test_strategy_in_workflow():
    on.workflow_dispatch()
    strategy.matrix(x=[1, 2, 3], y=["a", "b"], include={"z": 42})
    run(f"{matrix.x}, {matrix.y}, {matrix.z}")


@expect(
    """
# generated from test_workflow.py::test_matrix_from_input
on:
  workflow_dispatch:
    inputs:
      i:
        required: false
        type: string
jobs:
  test_matrix_from_input:
    runs-on: ubuntu-latest
    strategy:
      matrix: ${{ fromJson(inputs.i) }}
    steps:
    - run: ${{ matrix.foo }}, ${{ matrix.bar }}
    - name: Fail
      if: contains(inputs.i, 'failed')
"""
)
def test_matrix_from_input():
    on.workflow_dispatch()
    i = input()
    strategy.matrix(fromJson(i))
    run(f"{matrix.foo}, {matrix.bar}")
    step("Fail").if_(contains(i, "failed"))


@expect(
    """
# generated from test_workflow.py::test_runs_on_in_workflow
on:
  workflow_dispatch: {}
env:
  WORKFLOW_ENV: 1
jobs:
  test_runs_on_in_workflow:
    runs-on: macos-latest
    env:
      JOB_ENV: 2
"""
)
def test_runs_on_in_workflow():
    on.workflow_dispatch()
    env(WORKFLOW_ENV=1)
    runs_on("macos-latest")
    env(JOB_ENV=2)


@expect(
    """
# generated from test_workflow.py::test_runs_on_in_worfklow_with_name
name: Foo bar
on:
  workflow_dispatch: {}
jobs:
  test_runs_on_in_worfklow_with_name:
    name: Foo bar
    runs-on: macos-latest
"""
)
def test_runs_on_in_worfklow_with_name():
    name("Foo bar")
    on.workflow_dispatch()
    runs_on("macos-latest")


@expect(
    """
# generated from test_workflow.py::test_steps
on:
  workflow_dispatch: {}
jobs:
  my_job:
    runs-on: ubuntu-latest
    steps:
    - name: salutations
      run: echo hello
    - run: echo $WHO
      env:
        WHO: world
    - name: catastrophe
      if: failure()
      run: echo oh no
    - name: Checkout
      uses: actions/checkout@v4
      with:
        ref: dev
    - name: My action
      uses: ./my_action
      with:
        arg-1: foo
        arg_2: bar
        arg_3: baz
    - name: My other action
      uses: ./my_other_action
      with:
        arg-1: foo
        arg_2: bar
    - continue-on-error: true
      run: one
    - continue-on-error: value
      run: two
"""
)
def test_steps():
    on.workflow_dispatch()

    @job
    def my_job():
        step.run("echo hello").name("salutations")
        run("echo $WHO").env(WHO="world")
        step("catastrophe").run("echo oh no").if_("failure()")
        step.uses("actions/checkout@v4").with_(ref="dev")
        use("./my_action").with_(arg_1="foo", arg__2="bar").with_((("arg_3", "baz"),))
        use("./my_other_action", arg_1="foo", arg__2="bar")
        run("one").continue_on_error()
        run("two").continue_on_error("value")


@expect(
    """
# generated from test_workflow.py::test_workflow_dispatch_inputs
on:
  workflow_dispatch:
    inputs:
      foo:
        description: a foo
        required: true
        type: string
      bar:
        description: a bar
        required: false
        type: boolean
      baz:
        required: false
        default: b
        type: choice
        options:
        - a
        - b
        - c
      an-env:
        required: false
        type: environment
jobs:
  test_workflow_dispatch_inputs:
    runs-on: ubuntu-latest
    steps:
    - run: |
        echo ${{ inputs.foo }}
        echo ${{ inputs.bar }}
        echo ${{ inputs.baz }}
        echo ${{ inputs.an-env }}
"""
)
def test_workflow_dispatch_inputs():
    on.workflow_dispatch()
    foo = input.description("a foo").required()
    bar = input("a bar").type("boolean")
    baz = input.options("a", "b", "c").default("b")
    an_env = input.type("environment")
    run(f"""
        echo {foo}
        echo {bar}
        echo {baz}
        echo {an_env}
    """)  # fmt: skip


@expect(
    """
# generated from test_workflow.py::test_workflow_call
on:
  workflow_call:
    secrets:
      token:
        required: true
      auth:
        description: auth if provided
        required: false
    inputs:
      foo:
        required: true
        type: string
      bar:
        required: false
        type: boolean
      baz:
        required: false
        default: b
        type: choice
        options:
        - a
        - b
        - c
jobs:
  test_workflow_call:
    runs-on: ubuntu-latest
    steps:
    - run: |
        echo ${{ inputs.foo }}
        echo ${{ inputs.bar }}
        echo ${{ inputs.baz }}
"""
)
def test_workflow_call():
    on.workflow_call.secret("token", required=True).secret("auth", "auth if provided")

    foo = input.required()
    bar = input.type("boolean")
    baz = input.options("a", "b", "c").default("b")

    run(f"""
        echo {foo}
        echo {bar}
        echo {baz}
    """)  # fmt: skip


@expect(
    """
# generated from test_workflow.py::test_inputs
on:
  workflow_call:
    inputs:
      foo:
        description: a foo
        required: true
        type: string
      bar:
        required: false
        default: 42
        type: number
      baz:
        required: false
        default: true
        type: boolean
  workflow_dispatch:
    inputs:
      foo:
        description: a foo
        required: true
        type: string
      bar:
        required: false
        default: 42
        type: number
      baz:
        required: false
        default: true
        type: boolean
jobs:
  test_inputs:
    runs-on: ubuntu-latest
    steps:
    - run: |
        echo ${{ inputs.foo }}
        echo ${{ inputs.bar }} 
        echo ${{ inputs.baz }}
"""
)
def test_inputs():
    on.workflow_dispatch().workflow_call()
    foo = input("a foo").required()
    bar = input.default(42)
    baz = input.default(True)
    run(f"""
        echo {foo}
        echo {bar} 
        echo {baz}
    """)  # fmt: skip


@expect(
    """
# generated from test_workflow.py::test_input_underscores
on:
  workflow_dispatch:
    inputs:
      my-input:
        required: false
        type: string
      my_other_input:
        required: false
        type: string
      yet_another_input:
        required: false
        type: string
jobs:
  test_input_underscores:
    runs-on: ubuntu-latest
    steps:
    - run: echo ${{ inputs.my-input }} ${{ inputs.my_other_input }} ${{ inputs.yet_another_input
        }}
"""
)
def test_input_underscores():
    on.workflow_dispatch()
    my_input = input()
    my_other_input = input().id("my_other_input")
    yet__another__input = input()
    run(f"echo {my_input} {my_other_input} {yet__another__input}")


@expect(
    """
# generated from test_workflow.py::test_id
on: {}
jobs:
  test_id:
    runs-on: ubuntu-latest
    steps:
    - id: one
      run: one
    - id: y-1
      run: two
    - id: yy
      run: two prime
    - id: y
      run: three
    - name: use x
      run: ${{ steps.one.outputs }}
    - name: use y
      run: ${{ steps.y-1.outcome }}
    - name: use yy
      run: ${{ steps.yy.outputs.a }}
    - name: use z
      run: ${{ steps.y.result }}
    - id: step-1
      name: anon0
    - id: step-2
      name: anon1
    - id: step-3
      name: anon2
    - name: use anonymous
      run: |
        ${{ steps.step-1.outcome }}
        ${{ steps.step-2.outcome }}
        ${{ steps.step-3.outcome }}
"""
)
def test_id():
    x = step.id("one").run("one")
    y = step.run("two")
    yy = step.run("two prime").returns("a")
    z = step.id("y").run("three")

    step("use x").run(x.outputs)
    step("use y").run(y.outcome)
    step("use yy").run(yy.outputs.a)
    step("use z").run(z.result)

    code = "\n".join(str(step(f"anon{i}").outcome) for i in range(3))
    step("use anonymous").run(code)


@expect(
    """
# generated from test_workflow.py::test_step_id_underscores
on: {}
jobs:
  test_step_id_underscores:
    runs-on: ubuntu-latest
    steps:
    - id: my-step
      run: echo one
    - id: my_other_step
      run: echo two
    - id: yet_another_step
      run: echo three
    - if: steps.my-step && steps.my_other_step && steps.yet_another_step
      run: ''
"""
)
def test_step_id_underscores():
    my_step = run("echo one")
    my_other_step = run("echo two").id("my_other_step")
    yet__another__step = run("echo three")
    run("").if_(my_step & my_other_step & yet__another__step)


@expect(
    """
# generated from test_workflow.py::test_steps_array
on: {}
jobs:
  j:
    runs-on: ubuntu-latest
    steps:
    - name: ${{ steps.*.result }}
"""
)
def test_steps_array():
    @job
    def j():
        step(steps._.result)


@expect(
    """
# generated from test_workflow.py::test_if_expr
on: {}
jobs:
  test_if_expr:
    runs-on: ubuntu-latest
    steps:
    - id: x
      run: one
    - if: steps.x.outcome == 'success'
      run: two
    - if: '!steps.x.outputs'
      run: three
"""
)
def test_if_expr():
    x = step.run("one")
    step.run("two").if_(x.outcome == "success")
    step.run("three").if_(~x.outputs)


@expect(
    """
# generated from test_workflow.py::test_implicit_job_outputs
on: {}
jobs:
  j1:
    runs-on: ubuntu-latest
    outputs:
      one: ${{ steps.x.outputs.one }}
      two: ${{ steps.x.outputs.two }}
    steps:
    - id: x
      name: x
      run: |
        echo one=a >> $GITHUB_OUTPUTS
        echo two=b >> $GITHUB_OUTPUTS
  j2:
    runs-on: ubuntu-latest
    outputs:
      one: ${{ steps.x.outputs.one }}
      two: ${{ steps.x.outputs.two }}
      three: ${{ steps.y.outputs.three }}
      a: ${{ matrix.a }}
    strategy:
      matrix:
        a: [1, 2, 3]
    steps:
    - id: x
      name: x
      run: |
        echo one=a >> $GITHUB_OUTPUTS
        echo two=b >> $GITHUB_OUTPUTS
    - id: y
      name: y
      run: echo three=c >> $GITHUB_OUTPUTS
  j3:
    runs-on: ubuntu-latest
    outputs:
      one: ${{ steps.step-2.outputs.one }}
      two: ${{ steps.step-1.outputs.two }}
      a: ${{ matrix.a }}
    strategy:
      matrix:
        a: [1, 2, 3]
    steps:
    - id: step-1
      run: |
        echo one=a >> $GITHUB_OUTPUTS
        echo two=b >> $GITHUB_OUTPUTS
    - id: step-2
      run: echo one=c >> $GITHUB_OUTPUTS
"""
)
def test_implicit_job_outputs():
    @job
    def j1():
        x = step("x").returns(one="a", two="b")
        outputs(x)

    @job
    def j2():
        strategy.matrix(a=[1, 2, 3])
        x = step("x").returns(one="a", two="b")
        y = step("y").returns(three="c")
        outputs(x, y, matrix.a)

    @job
    def j3():
        strategy.matrix(a=[1, 2, 3])
        step.returns(one="a", two="b")
        step.returns(one="c")
        outputs("*", matrix.a)


@expect(
    """
# generated from test_workflow.py::test_explicit_job_outputs
on: {}
jobs:
  j:
    runs-on: ubuntu-latest
    outputs:
      foo: ${{ steps.x.outputs.one }}
      bar: ${{ steps.y.outputs.three }}
      baz: ${{ matrix.a }}
    strategy:
      matrix:
        a: [1, 2, 3]
    steps:
    - id: x
      name: x
      run: |
        echo one=a >> $GITHUB_OUTPUTS
        echo two=b >> $GITHUB_OUTPUTS
    - id: y
      name: y
      run: echo three=c >> $GITHUB_OUTPUTS
"""
)
def test_explicit_job_outputs():
    @job
    def j():
        strategy.matrix(a=[1, 2, 3])
        x = step("x").returns(one="a", two="b")
        y = step("y").returns(three="c")
        outputs(foo=x.outputs.one, bar=y.outputs.three, baz=matrix.a)


@expect(
    """
# generated from test_workflow.py::test_implicit_workflow_outputs
on:
  workflow_call:
    outputs:
      two:
        description: this is two
        value: ${{ jobs.j1.outputs.two }}
      one:
        value: ${{ jobs.j1.outputs.one }}
jobs:
  j1:
    outputs:
      one: 1
      two: 2
  j2:
    outputs:
      three: 3
"""
)
def test_implicit_workflow_outputs():
    on.workflow_call.output_descriptions(two="this is two")

    @job
    def j1():
        outputs(one=1, two=2)

    @job
    def j2():
        outputs(three=3)

    outputs(j1)


@expect(
    """
# generated from test_workflow.py::test_jolly_workflow_outputs
on:
  workflow_call:
    outputs:
      one:
        value: ${{ jobs.j1.outputs.one }}
      two:
        value: ${{ jobs.j1.outputs.two }}
      three:
        value: ${{ jobs.j2.outputs.three }}
jobs:
  j1:
    outputs:
      one: 1
      two: 2
  j2:
    outputs:
      three: 3
"""
)
def test_jolly_workflow_outputs():
    on.workflow_call()

    @job
    def j1():
        outputs(one=1, two=2)

    @job
    def j2():
        outputs(three=3)

    outputs("*")


@expect(
    """
# generated from test_workflow.py::test_needs
on: {}
jobs:
  j1: {}
  j2:
    needs: [j1]
    runs-on: ubuntu-latest
    steps:
    - if: needs.j1
      run: ''
  j3:
    needs: [j1, j2]
    runs-on: ubuntu-latest
    steps:
    # needs j1
    - run: ''
    # needs j1, j2
    - run: ''
  j4:
    needs: [j1, j3]
"""
)
def test_needs():
    @job
    def j1():
        pass

    @job
    def j2():
        run("").if_(j1)

    @job
    def j3():
        step.needs(j1).run("")
        step.needs(j1, j2).run("")

    @job
    def j4():
        needs(j1, j3)


@expect(
    """
# generated from test_workflow.py::test_job_as_context
on: {}
jobs:
  test_job_as_context:
    runs-on: ubuntu-latest
    steps:
    - run: 'false'
    - if: always()
      run: echo ${{ job.status }}
"""
)
def test_job_as_context():
    run("false")
    run(f"echo {job.status}").if_(always())


@expect(
    """
# generated from test_workflow.py::test_container
on: {}
jobs:
  j1:
    runs-on: ubuntu-latest
    container:
      image: node:18
      env:
        NODE_ENV: development
      ports:
      - 80
      volumes:
      - my_docker_volume:/volume_mount
      options:
      - --cpus 1
    steps:
    - run: echo ${{ job.container.id }}
  j2:
    container:
      image: ghcr.io/owner/image
      credentials:
        username: foo
        password: baz
"""
)
def test_container():
    @job
    def j1():
        container("node:18").env(NODE_ENV="development").ports([80])
        container.volumes(["my_docker_volume:/volume_mount"]).options(["--cpus 1"])
        run(f"echo {job.container.id}")

    @job
    def j2():
        container.image("ghcr.io/owner/image").credentials(
            username="foo", password="baz"
        )


@expect(
    """
# generated from test_workflow.py::test_services
on: {}
jobs:
  test_services:
    runs-on: ubuntu-latest
    services:
      nginx:
        image: nginx:latest
        ports:
        - 8080:80
      redis:
        ports:
        - 6379/tcp
    steps:
    - run: echo ${{ job.services.nginx.id }}
    - run: echo ${{ job.services.redis.ports[6379] }}
"""
)
def test_services():
    service("nginx", image="nginx:latest", ports=["8080:80"])
    service("redis", ports=["6379/tcp"])
    run(f"echo {job.services.nginx.id}")
    run(f"echo {job.services.redis.ports[6379]}")


@expect(
    """
# generated from test_workflow.py::test_strategy_as_context
on: {}
jobs:
  test_strategy_as_context:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        a: [1, 2, 3]
    steps:
    - run: |
        echo ${{ strategy }}
        echo ${{ strategy.fail-fast }}
        echo ${{ strategy.max-parallel }}
        echo ${{ strategy.job-index }}
        echo ${{ strategy.job-total }}
"""
)
def test_strategy_as_context():
    strategy.matrix(a=[1, 2, 3])
    run(f"""
        echo {strategy}
        echo {strategy.fail_fast}
        echo {strategy.max_parallel}
        echo {strategy.job_index}
        echo {strategy.job_total}
    """)  # fmt: skip


@expect(
    """
# generated from test_workflow.py::test_call
on: {}
jobs:
  j1:
    uses: foo
    with:
      arg-1: foo
      arg_2: bar
      arg_3: baz
  j2:
    uses: foo
    with:
      arg-1: foo
      arg_2: bar
"""
)
def test_call():
    @job
    def j1():
        call("foo")
        with_(arg_1="foo", arg__2="bar")
        with_((("arg_3", "baz"),))

    @job
    def j2():
        call("foo", arg_1="foo", arg__2="bar")
