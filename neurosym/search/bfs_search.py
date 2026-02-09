from typing import Iterable, TypeVar

from neurosym.search_graph.search_graph import SearchGraph

from .search_strategy import SearchStrategy

X = TypeVar("X")


class BFS(SearchStrategy):
    """
    Performs a breadth-first search on the given search graph, yielding each goal node
    in the order it was visited.

    :param iteration_limit: Maximum number of iterations to perform. Defaults to infinity.
    """

    def __init__(self, iteration_limit=float("inf")):
        self.iteration_limit = iteration_limit

    def search(self, graph: SearchGraph[X]) -> Iterable[X]:
        visited = set()
        queue = [graph.initial_node()]
        while queue:
            node = queue.pop(0)
            self.iteration_limit -= 1
            if self.iteration_limit < 0:
                break
            if node in visited:
                continue
            visited.add(node)
            yield from graph.yield_goal_node(node)
            for child in graph.expand_node(node):
                queue.append(child)
