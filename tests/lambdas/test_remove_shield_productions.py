import unittest

import neurosym as ns
from neurosym.examples import near


class TestRemoveShieldProductions(unittest.TestCase):
    def sanitize(self, expr_str):
        return ns.render_s_expression(
            near.with_shield.remove_shield_productions(ns.parse_s_expression(expr_str))
        )

    def test_no_shields(self):
        self.assertEqual(self.sanitize("(lam ($1))"), "(lam ($1))")

    def test_shield0_shifts_all(self):
        # If you shield 0, then all variables shift up by 1.
        self.assertEqual(self.sanitize("(lam (shield0 ($0)))"), "(lam ($1))")

    def test_shield0_shifts_higher(self):
        self.assertEqual(
            self.sanitize("(lam (shield0 (+ ($0) ($1) ($2) ($3))))"),
            "(lam (+ ($1) ($2) ($3) ($4)))",
        )

    def test_shield_above_variable(self):
        # If you shield 2, then all variables above 2 shift up by 1, but $0 and $1 stay the same.
        self.assertEqual(
            self.sanitize("(lam (shield1 (+ ($0) ($1) ($2) ($3))))"),
            "(lam (+ ($0) ($2) ($3) ($4)))",
        )

    def test_nested_shields(self):
        self.assertEqual(
            # shielding 1 within shielding 0 means shielding 2
            self.sanitize("(lam (shield0 (shield1 (+ ($0) ($1) ($2) ($3)))))"),
            "(lam (+ ($1) ($3) ($4) ($5)))",
        )

    def test_nested_shields_higher(self):
        self.assertEqual(
            self.sanitize("(lam (shield0 (shield1 (+ ($0) ($1)))))"),
            "(lam (+ ($1) ($3)))",
        )

    def test_shield_inside_function(self):
        self.assertEqual(
            self.sanitize("(lam (+ (shield0 ($0)) ($2)))"),
            "(lam (+ ($1) ($2)))",
        )

    def test_preserves_type_id(self):
        self.assertEqual(self.sanitize("(lam (shield0 ($0)))"), "(lam ($1))")

    def test_triple_nested(self):
        self.assertEqual(
            self.sanitize("(lam (shield0 (shield0 (shield0 ($0)))))"),
            "(lam ($3))",
        )

    def test_with_dispatch(self):
        self.assertEqual(
            self.sanitize("(dispatch (lam (shield0 (+ ($0) ($1)))))"),
            "(dispatch (lam (+ ($1) ($2))))",
        )
