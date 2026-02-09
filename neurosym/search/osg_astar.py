import queue
from dataclasses import dataclass, field
from types import NoneType
from typing import Iterable, Union

from neurosym.search.search_strategy import SearchStrategy
from neurosym.search_graph.search_graph import SearchGraph

X = type("X", (), {})


@dataclass(order=True)
class _OSGNode:
    cost: float
    node: X = field(compare=False)
    goal_results: list = field(default=None, compare=False)


class OSGAstar(SearchStrategy):
    """
    Performs an A* search on the given search graph with lazy cost evaluation.
    Children inherit their parent's cost when added to the fringe, and a node's
    own cost is only computed when it is popped. This avoids expensive cost
    computation for nodes that are never explored.

    Goal nodes are only yielded after being re-inserted with their true cost,
    ensuring they are returned in correct priority order.

    :param max_iterations: Maximum number of iterations to perform. Defaults to None.
    """

    def __init__(self, max_iterations: Union[int, NoneType] = None):
        self.max_iterations = max_iterations

    def search(self, graph: SearchGraph[X]) -> Iterable[X]:
        visited = set()
        fringe = queue.PriorityQueue()

        fringe.put(_OSGNode(0, graph.initial_node()))
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
            visited.add(entry.node)
            cost = graph.cost(entry.node)

            goal_results = list(graph.yield_goal_node(entry.node))
            if goal_results:
                fringe.put(_OSGNode(cost, entry.node, goal_results))

            for child in graph.expand_node(entry.node):
                fringe.put(_OSGNode(cost, child))
