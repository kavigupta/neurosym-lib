import heapq
from dataclasses import dataclass, field
from typing import Iterable, TypeVar

from neurosym.search_graph.search_graph import SearchGraph

from .search_strategy import SearchStrategy

X = TypeVar("X")


class AStar(SearchStrategy):
    """
    Performs an A* search on the given search graph, yielding each goal node in the
    order it was visited. Requires that the search graph implement a cost method.
    """

    def search(self, graph: SearchGraph[X]) -> Iterable[X]:
        visited = set()
        fringe = []

        def add_to_fringe(node):
            heapq.heappush(fringe, _AStarNode(graph.cost(node), node))

        add_to_fringe(graph.initial_node())
        # this is similar to the BFS algorithm
        # pylint: disable=duplicate-code
        while fringe:
            node = heapq.heappop(fringe).node
            if node in visited:
                continue
            visited.add(node)
            yield from graph.yield_goal_node(node)
            for child in graph.expand_node(node):
                add_to_fringe(child)


@dataclass(order=True)
class _AStarNode:
    """
    Represents a node in the A* search tree.
    """

    cost: float
    node: X = field(compare=False)
