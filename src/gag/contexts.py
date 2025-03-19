from .expr import contexts, RefExpr, Map, FlatMap


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


steps = _Contexts.steps
matrix = _Contexts.matrix
job = _Contexts.job
