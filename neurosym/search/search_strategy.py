from abc import ABC, abstractmethod
from typing import Iterator

from typing_extensions import TypeVar

from neurosym.search_graph.search_graph import SearchGraph

X = TypeVar("X")


class SearchStrategy(ABC):
    @abstractmethod
    def search(self, graph: SearchGraph[X]) -> Iterator[X]:
        """Perform a search based on the given query.

        Args:
            graph (SearchGraph): The search graph.

        Returns:
            Iterator: An iterator over search results.
        """
