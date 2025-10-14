import queue
from types import NoneType
from typing import Iterable, Union

from neurosym.search.astar_search import X, _AStarNode
from neurosym.search_graph.search_graph import SearchGraph


def osg_astar(
    g: SearchGraph[X], max_iterations: Union[int, NoneType] = None
) -> Iterable[X]:
    """
    Performs an A* search on the given search graph, yielding each goal node in the
    order it was visited. Requires that the search graph implement a cost method.

    :param g: Search graph to search over
    """
    visited = set()
    fringe = queue.PriorityQueue()

    def add_to_fringe(cost, node):
        fringe.put(_AStarNode(cost, node))

    add_to_fringe(0, g.initial_node())
    iterations = 0
    # this is similar to the BFS algorithm
    # pylint: disable=duplicate-code
    while not fringe.empty():
        if max_iterations is not None and iterations >= max_iterations:
            break
        iterations += 1
        node = fringe.get().node
        if node in visited:
            continue
        visited.add(node)
        cost = g.cost(node)
        yield from g.yield_goal_node(node)
        for child in g.expand_node(node):
            add_to_fringe(cost, child)
