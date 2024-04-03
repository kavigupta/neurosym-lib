import unittest

import numpy as np
import scipy.stats

import neurosym as ns


class ProbabilityTester(unittest.TestCase):
    def assertBinomial(self, n, p, pdelta, k):
        plow = p - pdelta
        phigh = p + pdelta
        kmin = int(plow * n)
        kmax = int(phigh * n)

        prob_mass = scipy.stats.binom.cdf(kmax, n, p) - scipy.stats.binom.cdf(
            kmin - 1, n, p
        )
        self.assertTrue(kmin > 0, f"Expected kmin > 0, got {kmin}")
        self.assertTrue(kmax < n - 1)
        self.assertTrue(
            prob_mass > 0.99, f"Insufficient probability mass: {prob_mass:.3f}"
        )
        self.assertTrue(kmin <= k <= kmax, f"Expected {k} to be in [{kmin}, {kmax}]")

    def assertSameProbDist(self, dist, target):
        print(dist.distribution.tolist())
        np.testing.assert_almost_equal(dist.distribution, target)


class ChildrenInOrderMask(ns.PreorderMask):
    def __init__(self, tree_dist, dsl):
        del dsl
        super().__init__(tree_dist)
        self.proper_context = False
        self.seen_symbols = []

    def compute_mask(self, position, symbols):
        if self.proper_context:
            mask = []
            for symbol in symbols:
                symbol = self.tree_dist.symbols[symbol][0]
                mask.append(int(symbol) == len(self.seen_symbols) + 1)
            return mask
        return [True] * len(symbols)

    def on_entry(self, position, symbol):
        symbol = self.tree_dist.symbols[symbol][0]
        if self.proper_context:
            self.seen_symbols.append(symbol)
        if symbol == "+":
            self.proper_context = True

    def on_exit(self, position, symbol):
        pass


class ChildrenInOrderAsserterMask(ns.PreorderMask):
    def __init__(self, tree_dist, dsl):
        del dsl
        super().__init__(tree_dist)
        self.proper_context = False
        self.seen_symbols = []

    def compute_mask(self, position, symbols):
        return [True] * len(symbols)

    def on_entry(self, position, symbol):
        symbol = self.tree_dist.symbols[symbol][0]
        if self.proper_context:
            self.seen_symbols.append(symbol)
            assert int(symbol) == len(
                self.seen_symbols
            ), f"Expected {len(self.seen_symbols)}, got {symbol}"
        if symbol == "+":
            self.proper_context = True

    def on_exit(self, position, symbol):
        pass
