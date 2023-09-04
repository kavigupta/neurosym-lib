from ..dsl.dsl_factory import DSLFactory


dslf = DSLFactory()
dslf.concrete("+", "(i, i) -> i", lambda x, y: x + y)
dslf.concrete("1", "() -> i", lambda: 1)
basic_arith_dsl = dslf.finalize()
