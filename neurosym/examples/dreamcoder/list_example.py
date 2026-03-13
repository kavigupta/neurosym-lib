from functools import reduce

from neurosym.dsl.dsl_factory import DSLFactory


def list_dslf(*output_types):
    """
    The List DSL from the DreamCoder repository (curried form).
    """
    dslf = DSLFactory(max_overall_depth=7)

    for i in range(6):
        dslf.production(str(i), "() -> i", lambda i=i: i)

    dslf.production("empty", "() -> [#T]", lambda: [])
    dslf.production("singleton", "#T -> [#T]", lambda x: [x])
    dslf.production("range", "i -> [i]", lambda x: list(range(x)))

    dslf.production("++", "[#T] -> [#T] -> [#T]", lambda x: lambda y: x + y)

    dslf.production(
        "mapi",
        "(i -> #T -> #R) -> [#T] -> [#R]",
        lambda f: lambda x: [f(i)(v) for i, v in enumerate(x)],
    )

    dslf.production(
        "reducei",
        "(i -> #R -> #T -> #R) -> #R -> [#T] -> #R",
        lambda f: lambda init: lambda xs: reduce(
            lambda acc, pair: f(pair[0])(acc)(pair[1]),
            enumerate(xs),
            init,
        ),
    )

    dslf.production("true", "() -> b", lambda: True)
    dslf.production("not", "b -> b", lambda x: not x)

    dslf.production("and", "b -> b -> b", lambda x: lambda y: x and y)
    dslf.production("or", "b -> b -> b", lambda x: lambda y: x or y)

    dslf.production(
        "i",
        "b -> #T -> #T -> #T",
        lambda cond: lambda y: lambda z: y if cond else z,
    )

    dslf.production("sort", "[#T] -> [#T]", sorted)

    dslf.production("+", "i -> i -> i", lambda x: lambda y: x + y)
    dslf.production("*", "i -> i -> i", lambda x: lambda y: x * y)

    dslf.production("negate", "i -> i", lambda x: -x)

    dslf.production("mod", "i -> i -> i", lambda x: lambda y: x % y)

    dslf.production("eq?", "i -> i -> b", lambda x: lambda y: x == y)
    dslf.production("gt?", "i -> i -> b", lambda x: lambda y: x > y)

    dslf.production(
        "is-prime",
        "i -> b",
        lambda x: x > 1 and all(x % i for i in range(2, min(1 + int(x**0.5), x))),
    )

    dslf.production(
        "is-square",
        "i -> b",
        lambda x: x > 1 and int(x**0.5) ** 2 == x,
    )

    dslf.production("sum", "[i] -> i", sum)

    dslf.production("reverse", "[#T] -> [#T]", lambda x: x[::-1])

    dslf.production(
        "all",
        "(#T -> b) -> [#T] -> b",
        lambda f: lambda xs: all(f(x) for x in xs),
    )

    dslf.production(
        "any",
        "(#T -> b) -> [#T] -> b",
        lambda f: lambda xs: any(f(x) for x in xs),
    )

    dslf.production(
        "index",
        "i -> [#T] -> #T",
        lambda i: lambda xs: xs[i],
    )

    dslf.production(
        "filter",
        "(#T -> b) -> [#T] -> [#T]",
        lambda f: lambda xs: [x for x in xs if f(x)],
    )

    dslf.production(
        "slice",
        "i -> i -> [#T] -> [#T]",
        lambda a: lambda b: lambda xs: xs[a:b],
    )

    dslf.no_zeroadic()

    dslf.lambdas(max_type_depth=3)
    dslf.prune_to(*output_types, prune_variables=False)
    return dslf


def list_dsl(*output_types):
    """The List DSL from the DreamCoder repository."""
    dslf = list_dslf(*output_types)
    return dslf.finalize()
