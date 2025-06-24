from ..dsl.dsl_factory import DSLFactory


def basic_arith_dsl(lambdas=False):
    """
    Create a DSL for basic arithmetic expressions, only including ``+`` and ``1``.

    :param lambdas: Whether to include lambda expressions in the DSL.
    """
    dslf = DSLFactory()
    dslf.concrete("+", "(i, i) -> i", lambda x, y: x + y)
    dslf.concrete("1", "() -> i", lambda: 1)
    if lambdas:
        dslf.lambdas()
    dslf.prune_to("i", "i -> i", "i -> i -> i", "(i, i) -> i", "(i, i) -> i -> i")
    return dslf.finalize()
