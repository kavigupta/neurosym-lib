from ..dsl.dsl_factory import DSLFactory


def basic_arith_dsl(lambdas=False):
    dslf = DSLFactory()
    dslf.concrete("+", "(i, i) -> i", lambda x, y: x + y)
    dslf.concrete("1", "() -> i", lambda: 1)
    if lambdas:
        dslf.lambdas()
    return dslf.finalize()
