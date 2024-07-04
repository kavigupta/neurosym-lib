import queue
from dataclasses import dataclass, field
from typing import Callable

from neurosym.programs.s_expression import SExpression
from neurosym.search_graph.search_graph import SearchGraph


def bounded_astar(
    g: SearchGraph,
    cost_plus_heuristic: Callable[[SExpression], float],
    max_depth: int,
):
    """
    Performs a bounded a-star search on the given search graph, yielding each goal node in
    the order it was visited. See ``astar`` for more details.

    :param g: Search graph to search over
    :param cost_plus_heuristic: An admissible cost heuristic.
    :param max_depth: Maximum depth to search to.
    :param depth_computer: Strategy to calculate program depth.
        Default strategy is to uniformly increment depth by one for each node.
    """
    assert max_depth > 0
    visited = set()
    fringe = queue.PriorityQueue()

    def add_to_fringe(node, depth):
        fringe.put(BoundedAStarNode(cost_plus_heuristic(node), node, depth))

    add_to_fringe(g.initial_node(), 0)
    while not fringe.empty():
        fringe_var = fringe.get()
        node, depth = fringe_var.node, fringe_var.depth
        if node.program in visited or depth > max_depth:
            continue
        visited.add(node.program)
        if g.is_goal_node(node):
            yield node
        for child in g.expand_node(node):
            add_to_fringe(child, depth + 1)


@dataclass(order=True)
class BoundedAStarNode:
    """
    Represents a node in the A* search tree.
    """

    cost: float
    node: SExpression = field(compare=False)
    depth: int = field(compare=True)
