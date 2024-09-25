from functools import lru_cache

from neurosym.dsl.dsl_factory import DSLFactory
from neurosym.examples.tower.state import TowerAction, TowerState
from neurosym.programs.s_expression import SExpression
from neurosym.programs.s_expression_render import parse_s_expression


@lru_cache(maxsize=None)
def tower_dsl():
    dslf = DSLFactory()
    for i in range(10):
        dslf.concrete(str(i), "() -> i", lambda *, i=i: i)
    dslf.concrete("+", "(i, i) -> i", lambda a, b: a + b)
    dslf.concrete("-", "(i, i) -> i", lambda a, b: a - b)
    dslf.concrete("*", "(i, i) -> i", lambda a, b: a * b)
    dslf.concrete("/", "(i, i) -> i", lambda a, b: a // b)

    dslf.concrete("l", "i -> t -> to", lambda n: lambda t: (t.left(n), []))
    dslf.concrete("r", "i -> t -> to", lambda n: lambda t: (t.right(n), []))
    for name, w, h in [("v", 1, 3), ("h", 3, 1), ("t", 1, 2), ("h2", 2, 1)]:
        dslf.concrete(name, "() -> t -> to", lambda *, w=w, h=h: TowerAction(0, w, h))

    def semi(a, b):
        def do(s):
            s, record_a = a(s)
            s, record_b = b(s)
            return s, record_a + record_b

        return do

    dslf.concrete("semi", "(t -> to, t -> to) -> t -> to", semi)

    # dslf.sugar("(/seq a b)", "(semi a b)")
    # dslf.sugar("(/seq a *b)", "(semi a (/seq *b))")

    def for_loop(n, f):
        def do(s):
            records = []
            for i in range(n):
                s, record = f(i)(s)
                records += record
            return s, records

        return do

    dslf.concrete("for", "(i, i -> t -> to) -> t -> to", for_loop)

    def embed(f):
        def do(s):
            return s, f(s)[1]

        return do

    dslf.concrete("embed", "(t -> to) -> t -> to", embed)
    dslf.lambdas()
    dslf.prune_to("t -> to")
    dsl = dslf.finalize()
    return dsl


def parse_sugared(s):
    """
    Parse s into an ns.SExpression, but with syntactic sugar that (/seq a b c) is transformed to (semi a (semi b c))
    """
    s = parse_s_expression(s)

    def expand_seq(s):
        children = [expand_seq(c) for c in s.children]
        if isinstance(s, SExpression) and s.symbol == "/seq":
            result = children[-1]
            for c in reversed(children[:-1]):
                result = SExpression("semi", [c, result])
            return result
        if isinstance(s, SExpression):
            return SExpression(s.symbol, children)
        return s

    return expand_seq(s)


def execute_tower(program):
    state, plan = tower_dsl().compute(tower_dsl().initialize(program))(TowerState())
    return state.hand, tuple(plan)
