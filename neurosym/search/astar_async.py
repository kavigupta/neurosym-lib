import heapq
import queue
from typing import Callable, Iterable, TypeVar

from pathos.multiprocessing import ProcessingPool as Pool

from neurosym.search.astar_search import _AStarNode
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


class AStarAsync(SearchStrategy):
    """
    Performs an A* search on the given search graph, yielding each node in
    the order it was visited. Evaluates the cost function asynchronously
    spawning max_workers processes. This function doesn't optimally traverse the search
    graph.

    :param max_workers: Maximum number of workers to use for evaluating
        the cost function.
    """

    def __init__(self, max_workers: int):
        assert max_workers > 0, "Cannot have 0 workers."
        self.max_workers = max_workers

    def search(self, graph: SearchGraph[X]) -> Iterable[X]:
        with Pool(self.max_workers) as executor:
            visited = set()
            fringe = _FuturePriorityQueue()

            def add_to_fringe(node):
                future = executor.apipe(graph.cost, node)
                fringe.putitem(
                    future,
                    future_fn=lambda ret: _AStarNode(ret, node),
                )

            add_to_fringe(graph.initial_node())
            while not fringe.empty():
                try:
                    fringe_var = fringe.getitem()
                    node = fringe_var.node
                    if node.program in visited:
                        continue
                    visited.add(node.program)
                    yield from graph.yield_goal_node(node)
                    for child in graph.expand_node(node):
                        add_to_fringe(child)
                except queue.Empty:
                    pass
