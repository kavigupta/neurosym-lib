from neurosym.dsl.dsl_factory import DSLFactory
from neurosym.types.type_string_repr import TypeDefiner


def _increment_and_return(counter):
    counter[0] += 1
    return counter[0]


def _create_mutable_arith_combinators_dsl():
    t = TypeDefiner()
    t.typedef("fn", "(i) -> i")

    dslf = DSLFactory()
    dslf.typedef("fn", "(i) -> i")
    dslf.concrete(
        "x",
        "() -> $fn",
        lambda: lambda t: t,
    )

    dslf.concrete(
        "1",
        "() -> $fn",
        lambda: lambda t: 1,
    )

    dslf.concrete(
        "+",
        "($fn, $fn) -> $fn",
        lambda x, y: lambda t: x(t) + y(t),
    )

    dslf.concrete(
        "*",
        "($fn, $fn) -> $fn",
        lambda x, y: lambda t: x(t) * y(t),
    )

    dslf.concrete(
        "even?",
        "($fn) -> $fn",
        lambda x: lambda t: x(t) % 2 == 0,
    )

    dslf.concrete(
        "ite",
        "($fn, $fn, $fn) -> $fn",
        lambda cond, fx, fy: lambda t: fx(t) if cond(t) else fy(t),
    )

    dslf.parameterized(
        "count",
        "() -> $fn",
        lambda counter: lambda t: _increment_and_return(counter),
        dict(counter=lambda: [0]),
    )
    return dslf.finalize()


# comment is in examples_other.rst
mutable_arith_combinators_dsl = _create_mutable_arith_combinators_dsl()
