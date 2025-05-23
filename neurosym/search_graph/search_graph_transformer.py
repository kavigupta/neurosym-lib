import itertools
from abc import abstractmethod
from typing import TypeVar

from .search_graph import SearchGraph

X = TypeVar("X")


class FilterEdgesGraph(SearchGraph[X]):
    """
    Abstract class for graphs that filter edges based on some criterion.
    """

    def __init__(self, graph: SearchGraph[X]):
        self.graph = graph

    @abstractmethod
    def include_edge(self, s, t) -> bool:
        """
        Returns True if the edge from s to t should be included in the graph.
        """

    def initial_node(self):
        return self.graph.initial_node()

    def expand_node(self, node):
        return [t for t in self.graph.expand_node(node) if self.include_edge(node, t)]

    def is_goal_node(self, node):
        return self.graph.is_goal_node(node)

    def cost(self, node):
        return self.graph.cost(node)

    def finalize(self, node) -> X:
        return self.graph.finalize(node)


class LimitEdgesGraph(SearchGraph[X]):
    """
    Limits the number of edges that can be expanded from a node, by only expanding the first
    ``limit`` edges.

    :param graph: The graph to limit the edges of.
    :param limit: The limit on the number of edges to expand.
    """

    def __init__(self, graph: SearchGraph[X], limit: int):
        self.graph = graph
        self.limit = limit

    def initial_node(self):
        return self.graph.initial_node()

    def expand_node(self, node):
        return itertools.islice(self.graph.expand_node(node), self.limit)

    def is_goal_node(self, node):
        return self.graph.is_goal_node(node)

    def cost(self, node):
        return self.graph.cost(node)

    def finalize(self, node) -> X:
        return self.graph.finalize(node)
