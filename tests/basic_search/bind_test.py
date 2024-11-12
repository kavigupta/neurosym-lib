import unittest

import neurosym as ns

dsl = ns.examples.basic_arith_dsl()


class StringContentsSearchGraph(ns.SearchGraph):
    def __init__(self, start, moves, character_to_match, desired_number):
        self.start = start
        self.moves = moves
        self.character_to_match = character_to_match
        self.desired_number = desired_number

    def initial_node(self):
        return self.start

    def is_goal_node(self, node: str) -> bool:
        return node.count(self.character_to_match) == self.desired_number

    def cost(self, node: str) -> int:
        return len(node) + abs(
            node.count(self.character_to_match) - self.desired_number
        )

    def expand_node(self, node):
        self.check_types_same(node, self.start)
        for move in self.moves:
            yield node + move

    def check_types_same(self, a, b):
        if isinstance(a, str) or isinstance(b, str):
            assert isinstance(a, str) and isinstance(b, str), (a, b)
            return
        assert isinstance(a, tuple) and isinstance(b, tuple), (a, b)
        all_items = a + b
        if not all_items:
            return
        for x in all_items:
            self.check_types_same(x, all_items[0])

    def finalize(self, node):
        return node


class TestSearch(unittest.TestCase):

    def test_astar_basic_1(self):

        g = StringContentsSearchGraph(
            start="",
            moves=["1A", "1B", "22A", "2B"],
            character_to_match="2",
            desired_number=4,
        )

        node = next(ns.search.astar(g))
        self.assertEqual(node, "22A22A")

    def test_astar_basic_2(self):
        g = StringContentsSearchGraph(
            start="",
            moves=["1A", "1BB", "2A", "2B"],
            character_to_match="B",
            desired_number=4,
        )
        node = next(ns.search.astar(g))
        self.assertEqual(node, "1BB1BB")

    def test_bind(self):
        g = ns.BindSearchGraph(
            StringContentsSearchGraph(
                start="",
                moves=["1A", "1B", "22A", "2B"],
                character_to_match="2",
                desired_number=4,
            ),
            lambda x: StringContentsSearchGraph(
                start=x,
                moves=["1A", "1BB", "2A", "2B"],
                character_to_match="B",
                desired_number=4,
            ),
        )
        node = next(ns.search.astar(g))
        # capable of backtracking slightly to find the optimal solution
        self.assertEqual(node, "2B2B2B2B")

    def test_bind_heterogenous_type(self):
        g = ns.BindSearchGraph(
            StringContentsSearchGraph(
                start="",
                moves=["1A", "1B", "22A", "2B"],
                character_to_match="2",
                desired_number=4,
            ),
            lambda x: StringContentsSearchGraph(
                start=(x,),
                moves=[("1",), ("2",)],
                character_to_match="2",
                desired_number=4,
            ),
        )
        node = next(ns.search.astar(g))
        self.assertEqual(node, ("22A22A", "2", "2", "2", "2"))

    def test_bind_three_types(self):
        g_1 = StringContentsSearchGraph(
            start="",
            moves=["1A", "1B", "22A", "2B"],
            character_to_match="2",
            desired_number=4,
        )
        g_2 = lambda x: StringContentsSearchGraph(
            start=(x,),
            moves=[("1",), ("2",)],
            character_to_match="2",
            desired_number=4,
        )
        g_3 = lambda x: StringContentsSearchGraph(
            start=(x,),
            moves=[(("1",),), (("2",),)],
            character_to_match=("1",),
            desired_number=4,
        )
        g = ns.BindSearchGraph(
            ns.BindSearchGraph(g_1, g_2),
            g_3,
        )
        node = next(ns.search.astar(g))
        self.assertEqual(
            node, (("22A22A", "2", "2", "2", "2"), ("1",), ("1",), ("1",), ("1",))
        )
