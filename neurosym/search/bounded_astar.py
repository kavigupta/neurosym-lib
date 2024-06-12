# pylint: skip-file
import os
import pickle
import queue
from dataclasses import dataclass, field
from typing import Callable, Tuple

from tqdm.auto import tqdm

from neurosym.programs.s_expression import SExpression
from neurosym.programs.s_expression_render import render_s_expression
from neurosym.search_graph.search_graph import SearchGraph


def bounded_astar(
    g: SearchGraph,
    cost_plus_heuristic: Callable[[SExpression], float],
    max_depth: int,
    verbose: bool = False,
):
    """
    Performs a bounded a-star search on the given search graph, yielding each node in
    the order it was visited.

    :param g: Search graph to search over
    :param cost_plus_heuristic: Cost plus heuristic function to use for A*.
        The heuristic function should be admissible, i.e. it should never overestimate
        the cost to reach the goal.
    :param max_depth: Maximum depth to search to.
    """
    assert max_depth > 0
    visited = set()
    fringe = queue.PriorityQueue()

    def add_to_fringe(node, depth):
        cost, metrics = cost_plus_heuristic(node)
        fringe.put(BoundedAStarNode(cost, node, depth, metrics))

    add_to_fringe(g.initial_node(), 0)
    best_node = None
    if verbose:
        pbar = tqdm(leave=False)
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
                f"Depth: {depth}, Cost: {cost:.4}, Program: {program:.50}, Acc: {program_acc:.4}, AUROC: {program_auroc:.4}"
            )
            pbar.update(1)

@dataclass(order=True)
class BoundedAStarNode:
    """
    Represents a node in the A* search tree.
    """

    cost: float
    node: SExpression = field(compare=False)
    depth: int = field(compare=True)
    # Accuracy, AUROC
    metrics: Tuple[float, float] = field(compare=False)
