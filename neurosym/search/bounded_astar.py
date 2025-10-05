import queue
from dataclasses import dataclass, field
from types import NoneType
from typing import Iterable, TypeVar, Union

from neurosym.programs.s_expression import SExpression
from neurosym.search_graph.search_graph import SearchGraph

X = TypeVar("X")


def bounded_astar(
    g: SearchGraph[X], max_depth: int, max_iterations: Union[int, NoneType] = None, frontier_capacity: int = 0
) -> Iterable[X]:
    """
    Performs a bounded a-star search on the given search graph, yielding each goal node in
    the order it was visited. See ``astar`` for more details.

    :param g: Search graph to search over
    :param max_depth: Maximum depth to search to.
    :param max_iterations: Maximum number of iterations to perform. If None, no limit is applied.
    :param frontier_capacity: Maximum number of nodes to keep in the fringe. If 0, no limit is applied.
    :return: An iterable of goal nodes in the order they were visited.
    
    :raises AssertionError: If `max_depth` is not greater than 0.
    """
    assert max_depth > 0
    visited = set()
    fringe = queue.PriorityQueue(maxsize=frontier_capacity)

    def add_to_fringe(node, depth):
        fringe.put(BoundedAStarNode(g.cost(node), node, depth))

    add_to_fringe(g.initial_node(), 0)
    iterations = 0
    while not fringe.empty():
        fringe_var = fringe.get()
        node, depth = fringe_var.node, fringe_var.depth
        if node in visited or depth > max_depth:
            continue
        visited.add(node)
        yield from g.yield_goal_node(node)
        for child in g.expand_node(node):
            add_to_fringe(child, depth + 1)
        iterations += 1
        if max_iterations is not None and iterations >= max_iterations:
            break


@dataclass(order=True)
class BoundedAStarNode:
    """
    Represents a node in the A* search tree.
    """

    cost: float
    node: SExpression = field(compare=False)
    depth: int = field(compare=True)
