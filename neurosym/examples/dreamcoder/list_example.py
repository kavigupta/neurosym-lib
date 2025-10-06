from functools import reduce

from neurosym.dsl.dsl_factory import DSLFactory


def list_dsl(*output_types):
    """
    The List DSL from the DreamCoder repository.
    """
    dslf = DSLFactory(max_overall_depth=5, max_expansion_steps=3)

    for i in range(6):
        dslf.production(str(i), "() -> i", lambda i=i: i)

    dslf.production("empty", "() -> [#T]", lambda: [])
    dslf.production("singleton", "#T -> [#T]", lambda x: [x])
    dslf.production("range", "i -> [i]", lambda x: list(range(x)))
    dslf.production("++", "([#T], [#T]) -> [#T]", lambda x, y: x + y)
    dslf.production(
        "mapi",
        "((i, #T) -> #R, [#T]) -> [#R]",
        lambda f: lambda x: [f(i, x) for i, x in enumerate(x)],
    )
    dslf.production(
        "reducei",
        "((i, #R, #T) -> #R, #R, [#T]) -> #R",
        lambda f: lambda x: lambda y: reduce(lambda x, y: f(i, x, y), x, y),
    )

    dslf.production("true", "() -> b", lambda: True)
    dslf.production("not", "b -> b", lambda x: not x)
    dslf.production("and", "(b, b) -> b", lambda x, y: x and y)
    dslf.production("or", "(b, b) -> b", lambda x, y: x or y)
    dslf.production("i", "(b, #T, #T) -> #T", lambda x, y, z: y if x else z)
    dslf.production("sort", "([#T]) -> [#T]", sorted)
    dslf.production("+", "(i, i) -> i", lambda x, y: x + y)
    dslf.production("*", "(i, i) -> i", lambda x, y: x * y)
    dslf.production("negate", "i -> i", lambda x: -x)
    dslf.production("mod", "(i, i) -> i", lambda x, y: x % y)
    dslf.production("eq?", "(i, i) -> b", lambda x, y: x == y)
    dslf.production("gt?", "(i, i) -> b", lambda x, y: x > y)
    dslf.production(
        "is-prime",
        "i -> b",
        lambda x: x > 1 and all(x % i for i in range(2, min(1 + int(x**0.5), x))),
    )
    dslf.production("is-square", "i -> b", lambda x: x > 1 and int(x**0.5) ** 2 == x)
    dslf.production("sum", "[i] -> i", sum)
    # # (lambda (lambda (reduce (lambda (lambda (+ $0 $1))) 0 $0)))
    dslf.production("reverse", "[#T] -> [#T]", lambda x: x[::-1])
    # (lambda (reduce (lambda (lambda (++ (singleton $0) $1))) empty $0))
    dslf.production(
        "all",
        "((#T) -> b, [#T]) -> b",
        lambda f: lambda x: all(f(i) for i in x),
    )
    # (lambda (lambda (reduce (lambda (lambda (and $0 $1))) false (map $1 $0))))
    dslf.production(
        "any",
        "((#T) -> b, [#T]) -> b",
        lambda f: lambda x: any(f(i) for i in x),
    )
    # (lambda (lambda (reduce (lambda (lambda (or $0 $1))) true (map $1 $0))))
    dslf.production("index", "(i, [#T]) -> #T", lambda x, y: y[x])
    # (lambda (lambda (reducei (lambda (lambda (lambda (if (eq? $1 $4) $0 0)))) 0 $0)))
    dslf.production(
        "filter",
        "((#T) -> b, [#T]) -> [#T]",
        lambda f: lambda x: [i for i in x if f(i)],
    )
    # (lambda (lambda
    #   (reduce
    #       (lambda (lambda (++ $1 (if ($3 $0) (singleton $0) empty))))
    #       empty
    #       $0)))
    dslf.production(
        "slice", "(i, i, [#T]) -> [#T]", lambda x: lambda y: lambda z: z[x:y]
    )
    # (lambda (lambda (lambda
    #   (reducei
    #     (lambda (lambda (lambda
    #         (++
    #             $2
    #             (if
    #                 (and
    #                     (or (gt? $1 $5) (eq? $1 $5))
    #                     (not (or (gt? $4 $1) (eq? $1 $4))))
    #                 (singleton $0)
    #                 empty)))))
    #     empty
    #     $0))))

    dslf.no_zeroadic()

    dslf.lambdas(max_type_depth=3)
    dslf.prune_to(*output_types, prune_variables=False)

    return dslf.finalize()
