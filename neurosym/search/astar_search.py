import queue
from dataclasses import dataclass, field
from typing import Iterable, TypeVar

from neurosym.search_graph.search_graph import SearchGraph

X = TypeVar("X")


def astar(g: SearchGraph[X]) -> Iterable[X]:
    """
    Performs an A* search on the given search graph, yielding each goal node in the
    order it was visited. Requires that the search graph implement a cost method.

    :param g: Search graph to search over
    """
    visited = set()
    fringe = queue.PriorityQueue()

    def add_to_fringe(node):
        fringe.put(_AStarNode(g.cost(node), node))

    add_to_fringe(g.initial_node())
    # this is similar to the BFS algorithm
    # pylint: disable=duplicate-code
    while not fringe.empty():
        node = fringe.get().node
        if node in visited:
            continue
        visited.add(node)
        yield from g.yield_goal_node(node)
        for child in g.expand_node(node):
            add_to_fringe(child)


@dataclass(order=True)
class _AStarNode:
    """
    Represents a node in the A* search tree.
    """

    cost: float
    node: X = field(compare=False)
