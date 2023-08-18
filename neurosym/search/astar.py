from dataclasses import dataclass, field
import queue
from typing import Callable
from neurosym.programs.s_expression import SExpression
from neurosym.search_graph.search_graph import SearchGraph


def astar(g: SearchGraph, cost_plus_heuristic: Callable[[SExpression], float]):
    """
    Performs a breadth-first search on the given search graph, yielding each node in the
    order it was visited.

    :param g: Search graph to search over
    :param cost_plus_heuristic: Cost plus heuristic function to use for A*.
        The heuristic function should be admissible, i.e. it should never overestimate
        the cost to reach the goal.
    """
    visited = set()
    fringe = queue.PriorityQueue()

    def add_to_fringe(node):
        fringe.put(AStarNode(cost_plus_heuristic(node), node))

    add_to_fringe(g.initial_node())
    while not fringe.empty():
        node = fringe.get().node
        if node in visited:
            continue
        visited.add(node)
        if g.is_goal_node(node):
            yield node
        for child in g.expand_node(node):
            add_to_fringe(child)


@dataclass(order=True)
class AStarNode:
    """
    Represents a node in the A* search tree.
    """

    cost: float
    node: SExpression = field(compare=False)
