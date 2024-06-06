import unittest
from fractions import Fraction
from typing import Any, Callable, Tuple

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

    def on_entry(self, position, symbol) -> Callable[[], None]:
        symbol = self.tree_dist.symbols[symbol][0]
        undos = []
        if self.proper_context:
            self.seen_symbols.append(symbol)
            undos.append(self.seen_symbols.pop)
        if symbol == "+":
            previous_context = self.proper_context
            self.proper_context = True
            undos.append(lambda: setattr(self, "proper_context", previous_context))
        return ns.chain_undos(undos)

    def on_exit(self, position, symbol) -> Callable[[], None]:
        return lambda: None

    @property
    def can_cache(self) -> bool:
        return False

    def cache_key(self, parents: Tuple[Tuple[int, int], ...]) -> Any:
        raise NotImplementedError


class ChildrenInOrderAsserterMask(ns.PreorderMask):
    def __init__(self, tree_dist, dsl):
        del dsl
        super().__init__(tree_dist)
        self.proper_context = False
        self.seen_symbols = []

    def compute_mask(self, position, symbols):
        return [True] * len(symbols)

    def on_entry(self, position, symbol) -> Callable[[], None]:
        symbol = self.tree_dist.symbols[symbol][0]
        undos = []
        if self.proper_context:
            self.seen_symbols.append(symbol)
            undos.append(self.seen_symbols.pop)
            assert int(symbol) == len(
                self.seen_symbols
            ), f"Expected {len(self.seen_symbols)}, got {symbol}"
        if symbol == "+":
            previous_context = self.proper_context
            self.proper_context = True
            undos.append(lambda: setattr(self, "proper_context", previous_context))
        return ns.chain_undos(undos)

    def on_exit(self, position, symbol):
        symbol = self.tree_dist.symbols[symbol][0]
        if symbol == "+":
            assert self.proper_context, "Expected proper context"
            assert (
                len(self.seen_symbols) == 3
            ), f"Expected 3 children, but received {len(self.seen_symbols)}"
            previous_context = self.proper_context
            previous_symbols = self.seen_symbols
            self.proper_context = False
            self.seen_symbols = []

            def undo():
                self.proper_context = previous_context
                self.seen_symbols = previous_symbols

            return undo
        return lambda: None

    @property
    def can_cache(self) -> bool:
        return False

    def cache_key(self, parents: Tuple[Tuple[int, int], ...]) -> Any:
        raise NotImplementedError


def enumerate_dsl(family, dist, min_likelihood=-6, max_denominator=10**6):
    result = list(family.enumerate(dist, min_likelihood=min_likelihood))
    result = [
        (
            ns.render_s_expression(prog),
            Fraction(*np.exp(likelihood).as_integer_ratio()).limit_denominator(
                max_denominator=max_denominator
            ),
        )
        for prog, likelihood in result
    ]
    result = sorted(result, key=lambda x: (-x[1], ns.render_s_expression(x[0])))
    result_display = str(result)
    print("{" + result_display[1:-1] + "}")
    return set(result)
