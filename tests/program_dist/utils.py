import unittest

import numpy as np
import scipy.stats


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
