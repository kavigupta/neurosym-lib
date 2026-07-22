import unittest

import neurosym as ns


def _polymorphic_list_dsl():
    """A tiny polymorphic DSL mirroring the DreamCoder list primitives that
    exposed the type-variable-capture bug: ``index``, ``++``, ``empty``,
    ``singleton``. Each polymorphic production declares a variable literally
    named ``#T``.
    """
    dslf = ns.DSLFactory(max_overall_depth=6)
    dslf.production("0", "() -> i", lambda: 0)
    dslf.production("empty", "() -> [#T]", lambda: [])
    dslf.production("singleton", "#T -> [#T]", lambda x: [x])
    dslf.production("++", "([#T], [#T]) -> [#T]", lambda a, b: a + b)
    dslf.production("index", "(i, [#T]) -> #T", lambda i, xs: xs[i])
    dslf.prune_to("[i]", "i")
    return dslf.finalize()


def _canonical_render(type_with_env):
    """Render a computed type with its type variables canonicalized to ``#t0``,
    ``#t1``, ... in order of first appearance, so assertions do not depend on the
    (globally-unique, intentionally non-deterministic) fresh variable names."""
    typ = type_with_env.typ
    subst = {}
    for node in typ.walk_type_nodes():
        if isinstance(node, ns.TypeVariable) and node.name not in subst:
            subst[node.name] = ns.TypeVariable(f"t{len(subst)}")
    return ns.render_type(typ.subst_type_vars(subst))


class TestPolymorphicComposition(unittest.TestCase):
    """Regression tests for type-variable capture across production instances.

    Every polymorphic production in a DSL may reuse the same variable name (here
    ``#T``), but each *use* of a production is an independent instance of its
    type scheme. Composing two such productions must not conflate their
    variables. See ``FunctionTypeSignature.unify_arguments``.
    """

    def setUp(self):
        self.dsl = _polymorphic_list_dsl()

    def _type(self, program):
        return self.dsl.compute_type(ns.parse_s_expression(program))

    def test_regression_baselines_still_work(self):
        # Both arguments of `++` share a type variable; concatenating two empty
        # lists must still type-check (this worked before the fix, by the
        # accident of shared variable names, and must keep working).
        self.assertEqual(_canonical_render(self._type("(++ (empty) (empty))")), "[#t0]")
        # `index` returns an element -- a bare type variable.
        self.assertEqual(_canonical_render(self._type("(index (0) (empty))")), "#t0")

    def test_element_flows_into_list_position(self):
        # `(++ (index 0 empty) empty)`: `index` returns an element whose type is
        # forced to be a list because it is an argument to `++`. This is well
        # typed -- `index`'s element type unifies with `[#t]` -- and DreamCoder
        # infers `int -> list(t)` for the enclosing abstraction. Before fresh
        # instantiation of each production's type scheme, the single shared `#T`
        # produced a spurious unification conflict and this returned None.
        result = self._type("(++ (index (0) (empty)) (empty))")
        self.assertIsNotNone(result)
        self.assertEqual(_canonical_render(result), "[#t0]")
        self.assertIsInstance(result.typ, ns.ListType)
        self.assertIsInstance(result.typ.element_type, ns.TypeVariable)

    def test_nested_polymorphic_application(self):
        # `(index 0 (index 0 empty))`: the inner `index` result is forced to be a
        # list by the outer `index`, whose result is a single element. Before the
        # fix the shared `#T` collapsed this to `[#T]` (an un-occurs-checked
        # cyclic `#T = [#T]` binding); it should be a bare element.
        result = self._type("(index (0) (index (0) (empty)))")
        self.assertEqual(_canonical_render(result), "#t0")
        self.assertIsInstance(result.typ, ns.TypeVariable)

    def test_singleton_baseline_matches(self):
        # Wrapping the element in a singleton first is the well-typed way to
        # write the same thing, and yields the same type.
        self.assertEqual(
            _canonical_render(
                self._type("(++ (singleton (index (0) (empty))) (empty))")
            ),
            "[#t0]",
        )


if __name__ == "__main__":
    unittest.main()
