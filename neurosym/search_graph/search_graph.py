from abc import ABC, abstractmethod
from typing import Callable, Generic, Iterable, TypeVar

N = TypeVar("N")
X = TypeVar("X")
Y = TypeVar("Y")


class SearchGraph(ABC, Generic[X]):
    """
    Represents a search graph where nodes are objects and edges are expansions of those objects.
    """

    @abstractmethod
    def initial_node(self) -> N:
        """
        The initial node of the search graph.
        """

    @abstractmethod
    def expand_node(self, node: N) -> Iterable[N]:
        """
        Find the neighbors of a node in the search graph.
        """

    @abstractmethod
    def is_goal_node(self, node: N) -> bool:
        """
        Return True iff the node is a goal node.
        """

    def cost(self, node: N) -> float:
        """
        The cost to reach the node from the initial node, plus a potential 'heuristic' cost
            (i.e., for A* search).
        """
        raise NotImplementedError(f"cost not implemented for {self.__class__.__name__}")

    @abstractmethod
    def finalize(self, node: N) -> X:
        """
        Finalize a goal node, returning the result of the search. This is useful for stripping
        out search graph metadata from the node.
        """

    def yield_goal_node(self, node: N) -> Iterable[X]:
        """
        Yield the final result of the search graph, if the node is a goal node.

        :param node: The node to check if it is a goal node.
        :return: Iterable of the final result of the search graph.
        """
        if self.is_goal_node(node):
            yield self.finalize(node)

    def bind(self, fn: Callable[[X, float], "SearchGraph[Y]"]) -> "SearchGraph[Y]":
        """
        Bind this search graph to another search graph. The nodes of this search graph will spawn
        the nodes of the new search graph.

        :param fn: A function that takes the final result of this search graph and returns a new
            search graph.
        """
        # pylint: disable=cyclic-import
        from .bind_search_graph import BindSearchGraph

        return BindSearchGraph(self, fn)

    def map(self, fn: Callable[[X], Y]) -> "SearchGraph[Y]":
        """
        Map the final result of the search graph to a new value.

        :param fn: A function that takes the final result of this search graph and returns a new
            result.
        """
        # pylint: disable=cyclic-import
        from .map_search_graph import MapSearchGraph

        return MapSearchGraph(self, fn)
