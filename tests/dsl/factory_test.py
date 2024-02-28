import unittest

import neurosym as ns
from ..utils import assertDSL


class TestPruning(unittest.TestCase):

    def test_basic_pruning(self):
        dslf = ns.DSLFactory()
        dslf.concrete("1", "() -> i", lambda x: x)
        dslf.concrete("identity", "i -> i", lambda x: x)
        dslf.concrete("add", "(i, i) -> i", lambda x, y: x + y)
        dslf.concrete("convert", "#x -> f", float)
        dslf.prune_to("f")
        dsl = dslf.finalize()
        assertDSL(
            self,
            dsl.render(),
            """
            1 :: () -> i
            add :: (i, i) -> i
            convert_0 :: f -> f
            convert_1 :: i -> f
            identity :: i -> i
            """,
        )

    def test_pruning_error(self):
        dslf = ns.DSLFactory()
        dslf.concrete("1", "() -> i", lambda x: x)
        dslf.concrete("identity", "i -> i", lambda x: x)
        dslf.concrete("add", "(i, i) -> i", lambda x, y: x + y)
        dslf.concrete("convert", "#x -> i", float)
        dslf.prune_to("f")
        self.assertRaisesRegex(
            TypeError,
            "All productions for .* were pruned. Check that the target types are correct.",
            dslf.finalize,
        )

    def test_pruning_specific(self):
        dslf = ns.DSLFactory()
        dslf.concrete("1", "() -> i", lambda x: x)
        dslf.concrete("identity", "f -> i", lambda x: x)
        dslf.concrete("add", "(i, i) -> i", lambda x, y: x + y)
        dslf.prune_to("i")
        self.assertRaises(
            TypeError,
            "All productions for identity were pruned. Check that the target types are correct.",
            dslf.finalize,
        )

    def test_allow_pruning(self):
        dslf = ns.DSLFactory()
        dslf.concrete("1", "() -> i", lambda x: x)
        dslf.concrete("identity", "f -> i", lambda x: x)
        dslf.concrete("add", "(i, i) -> i", lambda x, y: x + y)
        dslf.prune_to("i", tolerate_pruning_entire_productions=True)
        dsl = dslf.finalize()
        assertDSL(
            self,
            dsl.render(),
            """
            1 :: () -> i
            add :: (i, i) -> i
            """,
        )
