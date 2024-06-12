import unittest

import neurosym as ns

from .bigram_test import fam_with_ordering, fam_with_ordering_231, fam_with_vars


class TreeDistributionTest(unittest.TestCase):
    def collect_symbols(self, family, program):
        return ns.render_s_expression(
            ns.annotate_with_alternate_symbols(
                ns.parse_s_expression(program), family.tree_distribution_skeleton
            )
        )

    def test_basic_program(self):
        self.assertEqual(
            self.collect_symbols(fam_with_vars, "(+ (1) (1))"),
            "(+/+,1,2,call (1/+,1,2,call) (1/+,1,2,call))",
        )

    def test_program_with_variables(self):
        self.assertEqual(
            self.collect_symbols(fam_with_vars, "(call (lam ($0_0)) (1))"),
            "(call/+,1,2,call (lam/lam ($0_0/$0_0,+,1,2,call)) (1/+,1,2,call))",
        )

    def test_program_with_ordering(self):
        self.assertEqual(
            self.collect_symbols(fam_with_ordering, "(+ (1) (2) (3))"),
            "(+/+ (1/1) (2/2) (3/3))",
        )

    def test_program_with_ordering_231(self):
        self.assertEqual(
            self.collect_symbols(fam_with_ordering_231, "(+ (2) (3) (1))"),
            "(+/+ (2/2) (3/3) (1/1))",
        )
