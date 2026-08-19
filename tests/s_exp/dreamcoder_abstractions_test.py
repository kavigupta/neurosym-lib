import functools
import unittest

import neurosym as ns
from neurosym.compression.process_abstraction import (
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

    def test_parameter_as_lambda_body_does_not_crash(self):
        # Parameter $3 is the body of a nested inner lambda passed to mapi_1. A
        # naive implementation raises AssertionError in compute_type_abs. It is not
        # representable in the list DSL (mapi_1 wants an uncurried 2-argument
        # callback, DreamCoder supplies nested single-argument lambdas), so it must
        # be skipped (None) rather than crash.
        sig = self._signature_for(
            "#(lambda (lambda (lambda (mapi_1 (lambda (lambda $3)) (range $2)))))"
        )
        self.assertIsNone(sig)
