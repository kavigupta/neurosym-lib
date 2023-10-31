import neurosym as ns

from neurosym.near.search_graph import near_graph
from neurosym.search.bounded_astar import bounded_astar
from neurosym.types.type_string_repr import TypeDefiner


def assertDSLEnumerable(dsl, out_t, max_depth=5):
    t = TypeDefiner(L=10, O=4)
    t.typedef("fL", "{f, $L}")
    t.typedef("fO", "{f, $O}")

    def checker(x):
        """Initialize and return True always"""
        x = x.program
        xx = dsl.compute(dsl.initialize(x))
        print(xx)
        return True

    g = near_graph(dsl, t(out_t), is_goal=checker)

    def cost(x):
        if isinstance(x.program, ns.SExpression) and x.program.children:
            return len(str(x.program.children[0]))
        return 0

    # should not raise StopIteration.
    for _ in bounded_astar(g, cost, max_depth=max_depth):
        pass
