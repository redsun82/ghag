import threading

from .expr import contexts, RefExpr, Map, FlatMap
from .rules import *
from .workflow import *


@contexts
class _Contexts:
    class Steps(RefExpr):
        class Step(RefExpr):
            outputs: FlatMap
            result: RefExpr
            outcome: RefExpr

        __getattr__: Map[Step]

    steps: Steps
    matrix: FlatMap

    class Job(RefExpr):
        class Container(RefExpr):
            id: RefExpr
            network: RefExpr

        container: Container

        class Services(RefExpr):
            class Service(RefExpr):
                id: RefExpr
                network: RefExpr
                ports: RefExpr

            __getattr__: Map[Service]

        services: Services
        status: RefExpr

    job: Job

    class Jobs:
        class Job:
            outputs: FlatMap
            result: RefExpr

        __getattr__: Map[Job]

    jobs: Jobs
    needs: Jobs


steps = _Contexts.steps
matrix = _Contexts.matrix
job = _Contexts.job
jobs = _Contexts.jobs
needs = _Contexts.needs


@dataclasses.dataclass
class ContextBase(threading.local, RuleSet):
    current_workflow: Workflow | None = None
    current_job: Job | None = None
    current_workflow_id: str | None = None
    current_job_id: str | None = None

    def _knows_step_id(self, target: typing.Any, id: str) -> bool:
        if not self.current_job:
            return False
        for s in self.current_job.steps:
            if s is target:
                return False
            if s.id == id:
                return True
        return False

    def error(self, message: str):
        raise NotImplemented

    def check(self, cond: typing.Any, message: str) -> bool:
        if not cond:
            self.error(message)
        return cond

    @rule(steps)
    def v(self, *, target: typing.Any = None, field: str | None = None):
        return self.check(
            self.current_job,
            "`steps` can only be used in a job, did you forget a `@job` decoration?",
        ) and self.check(
            isinstance(target, Step) or field == "outputs",
            "`steps` can only be used while constructing a step or setting outputs",
        )

    @rule(steps._)
    def v(self, id: str, *, target: typing.Any = None, field: str | None = None):
        return self.check(
            self._knows_step_id(target, id),
            f"step `{id}` not defined yet in job `{self.current_job_id}`",
        )

    @rule(steps._.outputs._)
    def v(
        self,
        id: str,
        output: str,
        *,
        target: typing.Any = None,
        field: str | None = None,
    ):
        step = next(s for s in self.current_job.steps if s.id == id)
        return self.check(
            step.outputs and step.outputs and output in step.outputs,
            f"`{output}` was not declared in step `{id}`, use `returns()` declare it",
        )

    @rule(matrix)
    def v(self, *, target: typing.Any = None, field: str | None = None):
        print(target, field)
        return self.check(
            self.current_job
            and self.current_job.strategy is not None
            and self.current_job.strategy.matrix is not None,
            "`matrix` can only be used in a matrix job",
        ) and self.check(
            not isinstance(target, (Strategy, Matrix))
            or (isinstance(target, Job) and field == "strategy"),
            "`matrix` cannot be used in the `strategy` field defining it",
        )

    @rule(matrix._)
    def v(self, id, *, target: typing.Any = None, field: str | None = None):
        m = self.current_job.strategy.matrix
        # don't try to be smart if using something like an Expr
        return not isinstance(m, Matrix) or self.check(
            (m.values and id in m.values)
            or (m.include and any(id in include for include in m.include)),
            f"`{id}` was not declared in the `matrix` for this job",
        )

    @rule(job)
    def v(self, *, target: typing.Any = None, field: str | None = None):
        return self.check(self.current_job, "`job` can only be used in a job")

    @rule(job.container)
    def v(self, *, target: typing.Any = None, field: str | None = None):
        return self.check(
            self.current_job.container,
            "`job.container` can only be used in a containerized job",
        )

    @rule(job.services)
    def v(self, *, target: typing.Any = None, field: str | None = None):
        return self.check(
            self.current_job.services,
            "`job.services` can only be used in a job with services",
        )

    @rule(job.services._)
    def v(self, id, *, target: typing.Any = None, field: str | None = None):
        return self.check(
            id in self.current_job.services,
            f"no `{id}` service defined in `job.services`",
        )
