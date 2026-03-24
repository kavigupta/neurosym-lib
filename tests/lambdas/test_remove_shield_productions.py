import unittest

import neurosym as ns
from neurosym.examples.shield import remove_shield_productions


class TestRemoveShieldProductions(unittest.TestCase):
    def sanitize(self, expr_str):
        return ns.render_s_expression(
            remove_shield_productions(ns.parse_s_expression(expr_str))
        )

    def test_no_shields(self):
        self.assertEqual(self.sanitize("(lam ($1_0))"), "(lam ($1_0))")

    def test_shield0_shifts_all(self):
        # If you shield 0, then all variables shift up by 1.
        self.assertEqual(self.sanitize("(lam (shield0 ($0_0)))"), "(lam ($1_0))")

    def test_shield0_shifts_higher(self):
        self.assertEqual(
            self.sanitize("(lam (shield0 (+ ($0_0) ($1_0) ($2_0) ($3_0))))"),
            "(lam (+ ($1_0) ($2_0) ($3_0) ($4_0)))",
        )

    def test_shield_above_variable(self):
        # If you shield 2, then all variables above 2 shift up by 1, but $0_0 and $1_0 stay the same.
        self.assertEqual(
            self.sanitize("(lam (shield1 (+ ($0_0) ($1_0) ($2_0) ($3_0))))"),
            "(lam (+ ($0_0) ($2_0) ($3_0) ($4_0)))",
        )

    def test_nested_shields(self):
        self.assertEqual(
            # shielding 1 within shielding 0 means shielding 2
            self.sanitize("(lam (shield0 (shield1 (+ ($0_0) ($1_0) ($2_0) ($3_0)))))"),
            "(lam (+ ($1_0) ($3_0) ($4_0) ($5_0)))",
        )

    def test_nested_shields_higher(self):
        self.assertEqual(
            self.sanitize("(lam (shield0 (shield1 (+ ($0_0) ($1_0)))))"),
            "(lam (+ ($1_0) ($3_0)))",
        )

    def test_shield_inside_function(self):
        self.assertEqual(
            self.sanitize("(lam (+ (shield0 ($0_0)) ($2_0)))"),
            "(lam (+ ($1_0) ($2_0)))",
        )

    def test_preserves_type_id(self):
        self.assertEqual(self.sanitize("(lam (shield0 ($0_3)))"), "(lam ($1_3))")

    def test_triple_nested(self):
        self.assertEqual(
            self.sanitize("(lam (shield0 (shield0 (shield0 ($0_0)))))"),
            "(lam ($3_0))",
        )

    def test_with_dispatch(self):
        self.assertEqual(
            self.sanitize("(dispatch (lam (shield0 (+ ($0_0) ($1_0)))))"),
            "(dispatch (lam (+ ($1_0) ($2_0))))",
        )
