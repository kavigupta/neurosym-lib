import heapq
import queue
from typing import Callable, Iterable, TypeVar

from pathos.multiprocessing import ProcessingPool as Pool

from neurosym.search.bounded_astar import _BoundedAStarNode
from neurosym.search_graph.search_graph import SearchGraph

from .search_strategy import SearchStrategy

X = TypeVar("X")


class _FuturePriorityQueue:
    """
    A priority queue (heap-based) with primitive support for futures.
    """

    def __init__(self) -> None:
        self._heap: list = []
        self.waiting: list = []

    def syncronize(self):
        # move ready futures from waiting into the heap
        still_waiting = []
        for item, item_fn in self.waiting:
            if item.ready():
                heapq.heappush(self._heap, item_fn(item.get()))
            else:
                still_waiting.append((item, item_fn))
        self.waiting = still_waiting

    def putitem(self, item, future_fn: Callable):
        self.waiting.append((item, future_fn))
        self.syncronize()

    def empty(self):
        return len(self.waiting) == 0 and len(self._heap) == 0

    def getitem(self):
        self.syncronize()
        if not self._heap:
            raise queue.Empty
        return heapq.heappop(self._heap)


class BoundedAStarAsync(SearchStrategy):
    """
    Performs a bounded a-star search on the given search graph, yielding each node in
    the order it was visited. Evaluates the cost_plus_heuristic function asynchronously
    spawning max_workers processes. This function doesn't optimally traverse the search
    graph.

    :param max_depth: Maximum depth to search to.
    :param max_workers: Maximum number of workers to use for evaluating
        the cost_plus_heuristic function.
    """

    def __init__(self, max_depth: int, max_workers: int):
        assert max_depth > 0, "Cannot have 0 depth."
        assert max_workers > 0, "Cannot have 0 workers."
        self.max_depth = max_depth
        self.max_workers = max_workers

    def search(self, graph: SearchGraph[X]) -> Iterable[X]:
        with Pool(self.max_workers) as executor:
            visited = set()
            fringe = _FuturePriorityQueue()

            def add_to_fringe(node, depth):
                future = executor.apipe(graph.cost, node)
                fringe.putitem(
                    future,
                    future_fn=lambda ret: _BoundedAStarNode(ret, node, depth),
                )

            add_to_fringe(graph.initial_node(), 0)
            while not fringe.empty():
                try:
                    fringe_var = fringe.getitem()
                    # pylint: disable=duplicate-code
                    node, depth = fringe_var.node, fringe_var.depth
                    if node.program in visited or depth > self.max_depth:
                        continue
                    visited.add(node.program)
                    yield from graph.yield_goal_node(node)
                    for child in graph.expand_node(node):
                        add_to_fringe(child, depth + 1)
                except queue.Empty:
                    pass
