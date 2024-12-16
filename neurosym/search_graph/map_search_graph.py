from dataclasses import dataclass
from typing import Callable, TypeVar

from neurosym.search_graph.search_graph import SearchGraph

A = TypeVar("A")
B = TypeVar("B")


@dataclass
class MapSearchGraph(SearchGraph[B]):
    """
    Represents a search graph where the underlying graph's nodes are mapped over
    with a function.

    :param underlying_graph: The underlying search graph.
    :param map_fn: The function to map nodes with.
    """

    underlying_graph: SearchGraph[A]
    map_fn: Callable[[A], B]

    def initial_node(self):
        return self.underlying_graph.initial_node()

    def expand_node(self, node):
        return self.underlying_graph.expand_node(node)

    def is_goal_node(self, node):
        return self.underlying_graph.is_goal_node(node)

    def cost(self, node):
        return self.underlying_graph.cost(node)

    def finalize(self, node):
        return self.map_fn(self.underlying_graph.finalize(node))
