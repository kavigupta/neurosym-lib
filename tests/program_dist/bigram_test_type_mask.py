import unittest
from fractions import Fraction

import neurosym as ns

from .utils import enumerate_dsl


def get_dsl():
    dslf = ns.DSLFactory()
    dslf.production("+", "(i, i) -> i", lambda x, y: x + y)
    dslf.production("1", "() -> i", lambda: 1)
    dslf.production("singleton", "#a -> [#a]", lambda x: [x])
    dslf.production("head", "[#a] -> #a", lambda x: x[0])
    dslf.lambdas()
    dslf.prune_to("i -> i")
    return dslf.finalize()


dsl = get_dsl()


class BigramEnumerationTest(unittest.TestCase):

    def test_basic_type_mask(self):
        fam = ns.BigramProgramDistributionFamily(dsl)
        dist = fam.uniform()

        self.assertEqual(
            enumerate_dsl(fam, dist, -5),
            {
                ("(lam ($0_0))", Fraction(1, 8)),
                ("(lam (1))", Fraction(1, 8)),
                ("(head (singleton (lam ($0_0))))", Fraction(1, 32)),
                ("(head (singleton (lam (1))))", Fraction(1, 32)),
                ("(lam (head (singleton ($0_0))))", Fraction(1, 64)),
                ("(lam (head (singleton (1))))", Fraction(1, 64)),
                (
                    "(head (head (singleton (singleton (lam ($0_0))))))",
                    Fraction(1, 128),
                ),
                ("(head (head (singleton (singleton (lam (1))))))", Fraction(1, 128)),
                (
                    "(head (singleton (head (singleton (lam ($0_0))))))",
                    Fraction(1, 128),
                ),
                ("(head (singleton (head (singleton (lam (1))))))", Fraction(1, 128)),
                ("(lam (+ ($0_0) ($0_0)))", Fraction(1, 128)),
                ("(lam (+ ($0_0) (1)))", Fraction(1, 128)),
                ("(lam (+ (1) ($0_0)))", Fraction(1, 128)),
                ("(lam (+ (1) (1)))", Fraction(1, 128)),
            },
        )

    def test_elf_type_mask(self):
        fam = ns.BigramProgramDistributionFamily(
            dsl,
            include_type_preorder_mask=False,
            additional_preorder_masks=[ns.TypePreorderMaskELF],
        )
        dist = fam.uniform()

        self.assertEqual(
            enumerate_dsl(fam, dist, -5),
            {
                ("(lam ($0_0))", Fraction(1, 4)),
                ("(lam (1))", Fraction(1, 4)),
                ("(lam (head (singleton ($0_0))))", Fraction(1, 32)),
                ("(lam (head (singleton (1))))", Fraction(1, 32)),
                ("(lam (+ ($0_0) ($0_0)))", Fraction(1, 64)),
                ("(lam (+ ($0_0) (1)))", Fraction(1, 64)),
                ("(lam (+ (1) ($0_0)))", Fraction(1, 64)),
                ("(lam (+ (1) (1)))", Fraction(1, 64)),
                (
                    "(lam (head (head (singleton (singleton ($0_0))))))",
                    Fraction(1, 128),
                ),
                ("(lam (head (head (singleton (singleton (1))))))", Fraction(1, 128)),
            },
        )
