import queue
from dataclasses import dataclass, field
from typing import Iterable, TypeVar, Optional

from neurosym.programs.s_expression import SExpression
from neurosym.search_graph.search_graph import SearchGraph

from .search_strategy import SearchStrategy

X = TypeVar("X")


class BoundedAStar(SearchStrategy):
    """
    Performs a bounded a-star search on the given search graph, yielding each goal node in
    the order it was visited. See ``AStar`` for more details.

    :param max_depth: Maximum depth to search to.
    :param max_iterations: Maximum number of iterations to perform. Defaults to None
        (infinity).
    """

    def __init__(self, max_depth: int, max_iterations: Optional[int] = None):
        assert max_depth > 0
        self.max_depth = max_depth
        self.max_iterations = max_iterations

    def search(self, graph: SearchGraph[X]) -> Iterable[X]:
        visited = set()
        fringe = queue.PriorityQueue()

        def add_to_fringe(node, depth):
            fringe.put(BoundedAStarNode(graph.cost(node), node, depth))

        add_to_fringe(graph.initial_node(), 0)
        iterations = 0
        while not fringe.empty():
            fringe_var = fringe.get()
            node, depth = fringe_var.node, fringe_var.depth
            if node in visited or depth > self.max_depth:
                continue
            visited.add(node)
            yield from graph.yield_goal_node(node)
            for child in graph.expand_node(node):
                add_to_fringe(child, depth + 1)
            iterations += 1
            if self.max_iterations is not None and iterations >= self.max_iterations:
                break


@dataclass(order=True)
class BoundedAStarNode:
    """
    Represents a node in the A* search tree.
    """

    cost: float
    node: SExpression = field(compare=False)
    depth: int = field(compare=True)
