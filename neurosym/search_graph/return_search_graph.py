from typing import Iterable, TypeVar

from neurosym.search_graph.search_graph import SearchGraph

X = TypeVar("X")


class ReturnSearchGraph(SearchGraph[X]):
    """
    Search graph with exactly one goal node, which is passed in at initialization.
    """

    def __init__(self, result: X, cost: float):
        self.result = result
        self._cost = cost

    def initial_node(self) -> X:
        return self.result

    def expand_node(self, node: X) -> Iterable[X]:
        return []

    def is_goal_node(self, node: X) -> bool:
        assert node is self.result
        return True

    def cost(self, node: X) -> float:
        assert node is self.result
        return self._cost

    def finalize(self, node: X) -> X:
        return node
