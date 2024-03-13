"""
We want to enumerate programs as trees. Specifically, we have some distribution
    P(tree) = prod_{node in tree} P(node | ancestors(node)[:limit]).

Here, limit is a value that the distribution guarantees us. This is useful
    because we can use this to optimize our enumeration algorithm (since it
    reduces the amount of memory we need to save for each hole).

Our algorithm here is based on iterative deepening. We have a method that
    keeps a minimum and maximum likelihood, and enumerates all programs
    that are within this likelihood range.

Likelihood is defined as the log probability of the program.
"""

import itertools
from typing import Tuple

from neurosym.program_dist.enumeration_chunk_size import DEFAULT_CHUNK_SIZE
from neurosym.program_dist.tree_distribution.tree_distribution import TreeDistribution
from neurosym.programs.s_expression import SExpression


def enumerate_tree_dist(
    tree_dist: TreeDistribution,
    *,
    chunk_size: float = DEFAULT_CHUNK_SIZE,
    min_likelihood: float = float("-inf"),
):
    """
    Enumerate all programs using iterative deepening.

    Args:
        tree_dist: The distribution to sample from.
        chunk_size: The amount of likelihood to consider at once. If this is
            too small, we will spend a lot of time doing the same work over and
            over again. If this is too large, we will spend a lot of time
            doing work that we don't need to do.
    """
    for chunk in itertools.count(1):
        likelihood_bound = -chunk * chunk_size
        for program, likelihood in enumerate_tree_dist_dfs(
            tree_dist, likelihood_bound, ((0, 0),)
        ):
            if (
                max(likelihood_bound, min_likelihood)
                < likelihood
                <= likelihood_bound + chunk_size
            ):
                yield program, likelihood
        if likelihood_bound <= min_likelihood:
            return


def enumerate_tree_dist_dfs(
    tree_dist: TreeDistribution,
    min_likelihood: float,
    parents: Tuple[Tuple[int, int], ...],
):
    """
    Enumerate all programs that are within the likelihood range, with the given parents.
    """

    if min_likelihood > 0:
        # We can stop searching deeper.
        return

    assert len(parents) <= tree_dist.limit

    # Performed recursively for now.

    distribution = tree_dist.distribution[parents]
    for node, likelihood in distribution:
        symbol, arity = tree_dist.symbols[node]
        for children, child_likelihood in enumerate_children_and_likelihoods_dfs(
            tree_dist,
            min_likelihood - likelihood,
            parents,
            node,
            num_children=arity,
        ):
            yield SExpression(symbol, children), child_likelihood + likelihood


def enumerate_children_and_likelihoods_dfs(
    tree_dist: TreeDistribution,
    min_likelihood: float,
    parents: Tuple[Tuple[int, int], ...],
    most_recent_parent: int,
    num_children: int,
):
    """
    Enumerate all children and their likelihoods.
    """

    if num_children == 0:
        yield [], 0
        return

    new_parents = parents + ((most_recent_parent, num_children - 1),)
    new_parents = new_parents[-tree_dist.limit :]

    for last_child, last_likelihood in enumerate_tree_dist_dfs(
        tree_dist, min_likelihood, new_parents
    ):
        for rest_children, rest_likelihood in enumerate_children_and_likelihoods_dfs(
            tree_dist,
            min_likelihood - last_likelihood,
            parents,
            most_recent_parent,
            num_children - 1,
        ):
            yield rest_children + [last_child], last_likelihood + rest_likelihood
