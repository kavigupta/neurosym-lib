import queue
from typing import Callable

from pathos.multiprocessing import ProcessingPool as Pool

from neurosym.programs.s_expression import SExpression
from neurosym.search.bounded_astar import BoundedAStarNode
from neurosym.search_graph.search_graph import SearchGraph


class FuturePriorityQueue(queue.PriorityQueue):
    """
    A priority queue with primitive support for futures.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.waiting = []

    def syncronize(self):
        # check if any of the waiting items are done
        for i, (item, item_fn) in enumerate(self.waiting):
            if item.ready():
                # add to the queue
                self.put(item_fn(item.get()))
                # remove from waiting
                self.waiting.pop(i)

    def putitem(self, item, future_fn: Callable):
        """Making sure this doesn't override the `put` method."""
        self.waiting.append((item, future_fn))
        self.syncronize()

    def empty(self):
        return len(self.waiting) == 0 and super().empty()

    def getitem(self):
        """Making sure this doesn't override the `get` method."""
        self.syncronize()
        return self.get_nowait()


def bounded_astar_async(
    g: SearchGraph,
    cost_plus_heuristic: Callable[[SExpression], float],
    max_depth: int,
    max_workers: int,
):
    """
    Performs a bounded a-star search on the given search graph, yielding each node in
    the order it was visited. Evaluates the cost_plus_heuristic function asynchronously
    spawning max_workers processes. This function doesn't optimally traverse the search
    graph.

    :param g: Search graph to search over
    :param cost_plus_heuristic: Cost plus heuristic function to use for A*.
        The heuristic function should be admissible, i.e. it should never overestimate
        the cost to reach the goal.
    :param max_depth: Maximum depth to search to.
    :param max_workers: Maximum number of workers to use for evaluating
        the cost_plus_heuristic function.
    :param depth_computer: Strategy to calculate program depth.
        Default strategy is to uniformly increment depth by one for each node.
    """
    assert max_depth > 0, "Cannot have 0 depth."
    assert max_workers > 0, "Cannot have 0 workers."

    with Pool(max_workers) as executor:
        visited = set()
        fringe = FuturePriorityQueue()

        def add_to_fringe(node, depth):
            future = executor.apipe(cost_plus_heuristic, node)
            fringe.putitem(
                future,
                future_fn=lambda ret: BoundedAStarNode(ret, node, depth),
            )

        add_to_fringe(g.initial_node(), 0)
        while not fringe.empty():
            try:
                fringe_var = fringe.getitem()
                # pylint: disable=duplicate-code
                node, depth = fringe_var.node, fringe_var.depth
                if node.program in visited or depth > max_depth:
                    continue
                visited.add(node.program)
                if g.is_goal_node(node):
                    yield node
                for child in g.expand_node(node):
                    add_to_fringe(child, depth + 1)
            except queue.Empty:
                pass
