import unittest

import neurosym as ns

from .bind_test import simple_test_graph


class TestSearch(unittest.TestCase):

    def test_basic(self):

        g = simple_test_graph

        seen = []

        def map_fn(x):
            seen.append(x)
            return [x] * 2

        g = ns.MapSearchGraph(
            underlying_graph=g,
            map_fn=map_fn,
        )

        iterable = ns.search.astar(g)
        self.assertEqual(next(iterable), ["22A22A"] * 2)
        self.assertEqual(seen, ["22A22A"])
        self.assertEqual(next(iterable), ["2B22A2B"] * 2)
        self.assertEqual(seen, ["22A22A", "2B22A2B"])
        self.assertEqual(next(iterable), ["2B2B22A"] * 2)
        self.assertEqual(seen, ["22A22A", "2B22A2B", "2B2B22A"])
