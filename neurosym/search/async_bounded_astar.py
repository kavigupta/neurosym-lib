# pylint: skip-file
import os
import pickle
import queue
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Callable

from tqdm.auto import tqdm

from neurosym.programs.s_expression import SExpression
from neurosym.programs.s_expression_render import render_s_expression
from neurosym.search_graph.search_graph import SearchGraph
from neurosym.search.bounded_astar import BoundedAStarNode


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
            if item.done():
                # add to the queue
                self.put(item_fn(item.result()))
                # remove from waiting
                self.waiting.pop(i)

    def putitem(self, item: Future, future_fn: Callable):
        """Making sure this doesn't override the `put` method."""
        self.waiting.append((item, future_fn))
        self.syncronize()

    def empty(self):
        return len(self.waiting) == 0 and super().empty()

    def getitem(self):
        """Making sure this doesn't override the `get` method."""
        self.syncronize()
        return self.get_nowait()


def async_bounded_astar(
    g: SearchGraph,
    cost_plus_heuristic: Callable[[SExpression], float],
    max_depth: int,
    max_workers: int,
    verbose: bool = False,
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
    """
    assert max_depth > 0, "Cannot have 0 depth."
    assert max_workers > 0, "Cannot have 0 workers."

    with ProcessPoolExecutor(max_workers) as executor:
        visited = set()
        fringe = FuturePriorityQueue()

        def add_to_fringe(node, depth):
            future = executor.submit(cost_plus_heuristic, node)
            fringe.putitem(
                future, future_fn=lambda ret: BoundedAStarNode(ret[0], node, depth, ret[1])
            )

        add_to_fringe(g.initial_node(), 0)
        best_node = None
        if verbose:
            pbar = tqdm(leave=False)
        while not fringe.empty():
            try:
                fringe_var = fringe.getitem()
                node, depth = fringe_var.node, fringe_var.depth
                if node.program in visited or depth > max_depth:
                    continue
                visited.add(node.program)
                if g.is_goal_node(node):
                    yield node
                for child in g.expand_node(node):
                    add_to_fringe(child, depth + 1)

                if best_node is None or fringe_var.cost < best_node.cost:
                    best_node = fringe_var
                    # save the best node so far:
                    if not os.path.exists("list_best_nodes.pkl"):
                        with open("list_best_nodes.pkl", "wb") as f:
                            pickle.dump([], f)
                    
                    with open("list_best_nodes.pkl", "rb") as f:
                        best_nodes = pickle.load(f)
                    best_nodes.append(best_node)
                    with open("list_best_nodes.pkl", "wb") as f:
                        pickle.dump(best_nodes, f)

                if verbose:
                    depth = best_node.depth
                    cost = best_node.cost
                    program = render_s_expression(best_node.node.program)
                    program_acc = best_node.metrics[0]
                    program_auroc = best_node.metrics[1]
                    pbar.set_description(
                        f"Depth: {depth}, Cost: {cost:.4}, Program: {program:.50}, F1: {program_acc:.4}, AUROC: {program_auroc:.4}"
                    )
                    pbar.update(1)
            except queue.Empty:
                pass
