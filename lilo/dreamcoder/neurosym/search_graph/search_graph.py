from abc import ABC, abstractmethod
from typing import Iterable, TypeVar

N = TypeVar("N")


class SearchGraph(ABC):
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
