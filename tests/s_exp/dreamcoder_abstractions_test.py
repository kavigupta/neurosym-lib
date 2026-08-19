import functools
import types
import unittest

import neurosym as ns
from neurosym.compression.process_abstraction import (
    _convert_stitch_node,
    abstraction_production_from_dreamcoder,
)
from neurosym.examples.dreamcoder.list_example import list_dsl
from neurosym.types.type_string_repr import render_type

_LIST_OUTPUT_TYPES = [
    "[i] -> [i]", "[i] -> b", "[i] -> i", "b -> [i]", "b -> b",
    "b -> i", "i -> [i]", "i -> b", "i -> i",
]


@functools.lru_cache(maxsize=None)
def _list_dsl():
    # Building the list DSL is expensive (~20s); it is immutable and
    # abstraction_production_from_dreamcoder does not mutate it, so cache it.
    return list_dsl(*sorted(_LIST_OUTPUT_TYPES))


def _render_signature(signature):
    return render_type(signature.astype())


class TestAbstractionParameterTyping(unittest.TestCase):
    """Typing DreamCoder abstractions whose parameters are used at various depths
    and under various parents. An abstraction parameter is a stitch metavariable
    ``#k``; ``DSL.compute_type_abs`` must type it wherever it appears (including as
    the body of an inner lambda) and collect the parameters into the abstraction's
    argument list in application order.
    """

    def _signature_for(self, abstraction):
        prod = abstraction_production_from_dreamcoder(
            _list_dsl(), abstraction, abstraction
        )
        return None if prod is None else prod.type_signature()

    def test_single_parameter(self):
        sig = self._signature_for("#(lambda (gt? 1 (sum $0)))")
        self.assertEqual(_render_signature(sig), "[i] -> b")

    def test_two_parameters_ordered_by_application(self):
        # $1 is the first-applied argument (#0), $0 the second (#1); a naive
        # implementation both crashes here and reverses the argument order.
        sig = self._signature_for("#(lambda (lambda (gt? $1 (sum $0))))")
        self.assertEqual(_render_signature(sig), "(i, [i]) -> b")

    def test_parameter_as_lambda_body(self):
        # Parameter $3 is the body of a nested inner lambda passed to mapi_1. A
        # naive implementation raises AssertionError in compute_type_abs. The nested
        # single-argument lambdas are fused into mapi_1's 2-argument callback, so it
        # types: the callback ignores its args and returns parameter #1, mapping
        # (range #0) to a list of #1 -- i.e. (int, t) -> [t].
        sig = self._signature_for(
            "#(lambda (lambda (lambda (mapi_1 (lambda (lambda $3)) (range $2)))))"
        )
        self.assertIsNotNone(sig)
        self.assertEqual(len(sig.arguments), 2)
        self.assertEqual(render_type(sig.arguments[0]), "i")
        # return type is a list of the (polymorphic) second parameter.
        self.assertEqual(sig.return_type, ns.ListType(sig.arguments[1]))


class TestCurriedLambdaFusion(unittest.TestCase):
    """DreamCoder curries higher-order arguments as nested single-argument
    lambdas, but the list DSL types ``mapi``/``reducei`` with uncurried
    multi-argument arrows. The conversion fuses such lambda runs into the DSL's
    matching multi-argument lambda production; these tests pin the resulting types
    and the correctness of the de Bruijn remapping.
    """

    def _signature_for(self, abstraction):
        prod = abstraction_production_from_dreamcoder(
            _list_dsl(), abstraction, abstraction
        )
        return None if prod is None else prod.type_signature()

    def test_constant_callback(self):
        # mapi_1 with a callback that ignores both args and returns 0.
        sig = self._signature_for("#(lambda (mapi_1 (lambda (lambda 0)) $0))")
        self.assertEqual(_render_signature(sig), "[i] -> [i]")

    def test_callback_uses_both_fused_arguments(self):
        # (+ $1 $0) uses mapi_1's index ($1) and element ($0).
        sig = self._signature_for("#(lambda (mapi_1 (lambda (lambda (+ $1 $0))) $0))")
        self.assertEqual(_render_signature(sig), "[i] -> [i]")

    def test_three_argument_callback_reducei(self):
        # reducei_1 takes a 3-argument callback (index, acc, element) -> fused into
        # the DSL's 3-argument lambda (lam_2).
        sig = self._signature_for(
            "#(lambda (reducei_1 (lambda (lambda (lambda (+ $1 $0)))) 0 $0))"
        )
        self.assertEqual(_render_signature(sig), "[i] -> i")

    def test_fused_argument_order_is_not_symmetric(self):
        # mapi_3's callback binds (index : int, element : [i]). index-into-list
        # requires (int, list); it types only if the fused binders keep that order.
        ok = self._signature_for(
            "#(lambda (mapi_3 (lambda (lambda (index $1 $0))) $0))"
        )
        self.assertIsNotNone(ok)
        self.assertEqual(len(ok.arguments), 1)
        self.assertEqual(render_type(ok.arguments[0]), "[[i]]")
        # Swapping the two references is ill-typed (index wants int first).
        swapped = self._signature_for(
            "#(lambda (mapi_3 (lambda (lambda (index $0 $1))) $0))"
        )
        self.assertIsNone(swapped)

    def test_deeply_nested_variable_reference(self):
        # Inner mapi_1 callback references the OUTER mapi_3 index via $3, crossing
        # the fused inner group; exercises the depth-aware de Bruijn remap.
        sig = self._signature_for(
            "#(lambda (mapi_3 (lambda (lambda "
            "(mapi_1 (lambda (lambda (+ $3 $0))) $0))) $0))"
        )
        self.assertEqual(_render_signature(sig), "[[i]] -> [[i]]")


class TestNestedAbstractionReferences(unittest.TestCase):
    """One abstraction defined in terms of another arrives with the inner one
    inlined as `#(lambda ...)`. Once that inner abstraction is a production in the
    DSL, the outer one references it rather than being skipped.
    """

    def test_references_added_production(self):
        inner = "#(lambda (gt? 1 (sum $0)))"
        outer = "#(lambda (#(lambda (gt? 1 (sum $0))) $0))"  # outer applies inner

        dsl = _list_dsl()
        # Before the inner abstraction is added, the outer one cannot be represented.
        self.assertIsNone(
            abstraction_production_from_dreamcoder(dsl, outer, outer)
        )

        inner_prod = abstraction_production_from_dreamcoder(dsl, inner, inner)
        dsl = dsl.add_productions(inner_prod)
        outer_prod = abstraction_production_from_dreamcoder(dsl, outer, outer)
        # Outer applies inner (an `[i] -> b`) to its own `[i]` argument, so it has
        # the same type, and its body references the inner production's symbol.
        self.assertEqual(
            render_type(outer_prod.type_signature().astype()), "[i] -> b"
        )
        self.assertIn(inner, ns.render_s_expression(outer_prod._body))

    def test_reference_inside_fused_lambda(self):
        # The referenced abstraction can appear inside a fused multi-arg lambda,
        # applied to a remapped bound variable.
        inner = "#(lambda (gt? 1 (sum $0)))"
        outer = (
            "#(lambda (mapi_3 (lambda (lambda "
            "(#(lambda (gt? 1 (sum $0))) $0))) $0))"
        )
        dsl = _list_dsl()
        dsl = dsl.add_productions(
            abstraction_production_from_dreamcoder(dsl, inner, inner)
        )
        outer_prod = abstraction_production_from_dreamcoder(dsl, outer, outer)
        self.assertEqual(
            render_type(outer_prod.type_signature().astype()), "[[i]] -> [b]"
        )


class TestSkipReasonsAndAssertions(unittest.TestCase):
    """Cases that cannot occur for valid input (given how Stitch is invoked) must
    fail loudly with an assertion rather than being silently skipped.
    """

    def test_free_variable_asserts(self):
        # DreamCoder abstractions are closed and every parameter reference is
        # rewritten to `#k` upstream, so a surviving free `$` is impossible for
        # valid input -- it signals malformed input or a binder-tracking bug.
        node = types.SimpleNamespace(
            symbol="sum",
            children=(types.SimpleNamespace(symbol="$5", children=()),),
        )
        with self.assertRaises(AssertionError):
            _convert_stitch_node(node, None, _list_dsl(), {}, [], 0)

    def test_applied_metavariable_asserts(self):
        # A metavariable applied to arguments (higher-order abstraction) is
        # forbidden by Stitch's no_curried_metavars flag. Stitch renders it with a
        # bare `#` head and a non-lambda first child (e.g. `#(1)` for `($0 1)`),
        # distinct from a nested `#(lambda ...)` invention.
        node = types.SimpleNamespace(
            symbol="#",
            children=(types.SimpleNamespace(symbol="1", children=()),),
        )
        with self.assertRaises(AssertionError):
            _convert_stitch_node(node, None, _list_dsl(), {}, [], 0)

    def test_partially_applied_higher_order_argument_asserts(self):
        # mapi_1's slot expects a 2-argument callback (arity 2); supplying only one
        # lambda cannot happen under Stitch's eta_long flag, so assert.
        one_lambda = types.SimpleNamespace(
            symbol="lambda",
            children=(types.SimpleNamespace(symbol="0", children=()),),
        )
        with self.assertRaises(AssertionError):
            _convert_stitch_node(one_lambda, 2, _list_dsl(), {}, [], 0)
