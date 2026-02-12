import queue
from dataclasses import dataclass, field
from typing import Iterable, Optional, TypeVar

from neurosym.search.search_strategy import SearchStrategy
from neurosym.search_graph.search_graph import SearchGraph

X = TypeVar("X")


@dataclass(order=True)
class _OSGNode:
    cost: float
    node: X = field(compare=False)
    depth: int = field(default=0, compare=False)
    goal_results: list = field(default=None, compare=False)


class OSGAstar(SearchStrategy):
    """
    Performs an A* search on the given search graph with lazy cost evaluation.
    Children inherit their parent's cost when added to the fringe, and a node's
    own cost is only computed when it is popped. This avoids expensive cost
    computation for nodes that are never explored.

    Goal nodes are only yielded after being re-inserted with their true cost,
    ensuring they are returned in correct priority order.

    :param max_depth: Maximum search depth. Defaults to None (no limit).
    :param max_iterations: Maximum number of iterations to perform. Defaults to None.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        max_iterations: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.max_iterations = max_iterations

    def search(self, graph: SearchGraph[X]) -> Iterable[X]:
        visited = set()
        fringe = queue.PriorityQueue()

        fringe.put(_OSGNode(0, graph.initial_node(), 0))
        iterations = 0
        while not fringe.empty():
            if self.max_iterations is not None and iterations >= self.max_iterations:
                break
            iterations += 1
            entry = fringe.get()

            if entry.goal_results is not None:
                yield from entry.goal_results
                continue

            if entry.node in visited:
                continue
            if self.max_depth is not None and entry.depth > self.max_depth:
                continue
            visited.add(entry.node)
            cost = graph.cost(entry.node)

            goal_results = list(graph.yield_goal_node(entry.node))
            if goal_results:
                fringe.put(_OSGNode(cost, entry.node, entry.depth, goal_results))

            for child in graph.expand_node(entry.node):
                fringe.put(_OSGNode(cost, child, entry.depth + 1))
