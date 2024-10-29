import neurosym as ns
from neurosym.examples import near


def assertDSLEnumerable(dsl, out_t, max_depth=5):
    t = ns.TypeDefiner(L=10, O=4)
    t.typedef("fL", "{f, $L}")
    t.typedef("fO", "{f, $O}")

    def checker(x):
        """Initialize and return True always"""
        x = x.program
        dsl.compute(dsl.initialize(x))
        return True

    g = near.near_graph(dsl, t(out_t), max_depth=max_depth, is_goal=checker)

    def cost(x):
        if isinstance(x.program, ns.SExpression) and x.program.children:
            return len(str(x.program.children[0]))
        return 0

    # should not raise StopIteration.
    for _ in ns.search.bounded_astar(g, cost, max_depth=max_depth):
        pass
