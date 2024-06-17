from functools import reduce

import numpy as np
from permacache import permacache

from neurosym.datasets.load_data import get_raw_url, load_json
from neurosym.dsl.dsl import DSL
from neurosym.dsl.dsl_factory import DSLFactory
from neurosym.examples.dreamcoder.dreamcoder import Domain, IOExample, Task
from neurosym.programs.s_expression import SExpression


def list_dsl(*output_types):
    """
    The List DSL from the DreamCoder repository.
    """
    dslf = DSLFactory(max_overall_depth=5, max_expansion_steps=3)

    for i in range(6):
        dslf.concrete(str(i), "() -> i", lambda i=i: i)

    dslf.concrete("empty", "() -> [#T]", lambda: [])
    dslf.concrete("singleton", "#T -> [#T]", lambda x: [x])
    dslf.concrete("range", "i -> [i]", lambda x: list(range(x)))
    dslf.concrete("++", "([#T], [#T]) -> [#T]", lambda x, y: x + y)
    dslf.concrete(
        "mapi",
        "((i, #T) -> #R, [#T]) -> [#R]",
        lambda f: lambda x: [f(i, x) for i, x in enumerate(x)],
    )
    dslf.concrete(
        "reducei",
        "((i, #R, #T) -> #R, #R, [#T]) -> #R",
        lambda f: lambda x: lambda y: reduce(lambda x, y: f(i, x, y), x, y),
    )

    dslf.concrete("true", "() -> b", lambda: True)
    dslf.concrete("not", "b -> b", lambda x: not x)
    dslf.concrete("and", "(b, b) -> b", lambda x, y: x and y)
    dslf.concrete("or", "(b, b) -> b", lambda x, y: x or y)
    dslf.concrete("i", "(b, #T, #T) -> #T", lambda x, y, z: y if x else z)
    dslf.concrete("sort", "([#T]) -> [#T]", sorted)
    dslf.concrete("+", "(i, i) -> i", lambda x, y: x + y)
    dslf.concrete("*", "(i, i) -> i", lambda x, y: x * y)
    dslf.concrete("negate", "i -> i", lambda x: -x)
    dslf.concrete("mod", "(i, i) -> i", lambda x, y: x % y)
    dslf.concrete("eq?", "(i, i) -> b", lambda x, y: x == y)
    dslf.concrete("gt?", "(i, i) -> b", lambda x, y: x > y)
    dslf.concrete(
        "is-prime",
        "i -> b",
        lambda x: x > 1 and all(x % i for i in range(2, min(1 + int(x**0.5), x))),
    )
    dslf.concrete("is-square", "i -> b", lambda x: x > 1 and int(x**0.5) ** 2 == x)
    dslf.concrete("sum", "[i] -> i", sum)
    # # (lambda (lambda (reduce (lambda (lambda (+ $0 $1))) 0 $0)))
    dslf.concrete("reverse", "[#T] -> [#T]", lambda x: x[::-1])
    # (lambda (reduce (lambda (lambda (++ (singleton $0) $1))) empty $0))
    dslf.concrete(
        "all",
        "((#T) -> b, [#T]) -> b",
        lambda f: lambda x: all(f(i) for i in x),
    )
    # (lambda (lambda (reduce (lambda (lambda (and $0 $1))) false (map $1 $0))))
    dslf.concrete(
        "any",
        "((#T) -> b, [#T]) -> b",
        lambda f: lambda x: any(f(i) for i in x),
    )
    # (lambda (lambda (reduce (lambda (lambda (or $0 $1))) true (map $1 $0))))
    dslf.concrete("index", "(i, [#T]) -> #T", lambda x, y: y[x])
    # (lambda (lambda (reducei (lambda (lambda (lambda (if (eq? $1 $4) $0 0)))) 0 $0)))
    dslf.concrete(
        "filter",
        "((#T) -> b, [#T]) -> [#T]",
        lambda f: lambda x: [i for i in x if f(i)],
    )
    # (lambda (lambda
    #   (reduce
    #       (lambda (lambda (++ $1 (if ($3 $0) (singleton $0) empty))))
    #       empty
    #       $0)))
    dslf.concrete("slice", "(i, i, [#T]) -> [#T]", lambda x: lambda y: lambda z: z[x:y])
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


@permacache("neurosym/examples/dreamcoder/load_dreamcoder_task")
def load_dreamcoder_task(name):
    """
    Load a task from the DreamCoder repository.
    """
    return load_json(
        get_raw_url("https://github.com/ellisk42/ec/tree/master/data", f"{name}.json")
    )


def dreamcoder_list_tasks():
    """
    The List tasks from the DreamCoder repository.
    """
    return load_dreamcoder_task("list_tasks")


def dreamcoder_list_tasks_2():
    """
    The List tasks from the DreamCoder repository.
    """
    return load_dreamcoder_task("list_tasks2")


class ListDomain(Domain):

    def io_log_prob(
        self, dsl: DSL, program: SExpression, io_example: IOExample
    ) -> float:
        correct = dsl.compute(program)(io_example.inputs) == io_example.output
        if correct:
            return 0
        return float("-inf")

    def sample_task(
        self, dsl: DSL, program: SExpression, rng: np.random.RandomState
    ) -> Task:
        typ = dsl.compute_type(program)
        raise NotImplementedError(f"Sampling tasks for {typ} not implemented.")
