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

import copy
import itertools
from typing import List, Tuple

import numpy as np

from neurosym.program_dist.enumeration_chunk_size import DEFAULT_CHUNK_SIZE
from neurosym.program_dist.tree_distribution.preorder_mask.preorder_mask import (
    PreorderMask,
)
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
        preorder_mask = tree_dist.mask_constructor(tree_dist)
        preorder_mask.on_entry(0, 0)
        for program, likelihood, _ in enumerate_tree_dist_dfs(
            tree_dist, likelihood_bound, ((0, 0),), preorder_mask
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
    preorder_mask: PreorderMask,
):
    """
    Enumerate all programs that are within the likelihood range, with the given parents.
    """

    if min_likelihood > 0:
        # We can stop searching deeper.
        return

    assert len(parents) <= tree_dist.limit
    assert isinstance(
        preorder_mask, PreorderMask
    ), f"{preorder_mask} is not a PreorderMask"

    # Performed recursively for now.
    syms, log_probs = tree_dist.likelihood_arrays[parents]
    position = parents[-1][1]
    mask = preorder_mask.compute_mask(position, syms)
    denominator = np.logaddexp.reduce(log_probs[mask])
    for node, likelihood in zip(syms[mask], log_probs[mask] - denominator):
        preorder_mask_copy = copy.deepcopy(preorder_mask)
        new_parents = parents + (node,)
        new_parents = new_parents[-tree_dist.limit :]
        symbol, arity = tree_dist.symbols[node]
        preorder_mask_copy.on_entry(position, node)
        for (
            children,
            child_likelihood,
            preorder_mask_copy,
        ) in enumerate_children_and_likelihoods_dfs(
            tree_dist,
            min_likelihood - likelihood,
            parents,
            node,
            num_children=arity,
            starting_index=0,
            order=tree_dist.ordering.order(node, arity),
            preorder_mask=preorder_mask_copy,
        ):
            preorder_mask_copy.on_exit(position, node)
            yield SExpression(
                symbol, [children[i] for i in range(arity)]
            ), child_likelihood + likelihood, preorder_mask_copy


def enumerate_children_and_likelihoods_dfs(
    tree_dist: TreeDistribution,
    min_likelihood: float,
    parents: Tuple[Tuple[int, int], ...],
    most_recent_parent: int,
    num_children: int,
    starting_index: int,
    order: List[int],
    preorder_mask: PreorderMask,
):
    """
    Enumerate all children and their likelihoods.
    """

    if starting_index == num_children:
        yield {}, 0, preorder_mask
        return
    new_parents = parents + ((most_recent_parent, order[starting_index]),)
    new_parents = new_parents[-tree_dist.limit :]

    for first_child, first_likelihood, preorder_mask_2 in enumerate_tree_dist_dfs(
        tree_dist, min_likelihood, new_parents, preorder_mask
    ):
        for (
            rest_children,
            rest_likelihood,
            preorder_mask_3,
        ) in enumerate_children_and_likelihoods_dfs(
            tree_dist,
            min_likelihood - first_likelihood,
            parents,
            most_recent_parent,
            num_children,
            starting_index + 1,
            order,
            preorder_mask_2,
        ):
            yield {
                order[starting_index]: first_child,
                **rest_children,
            }, first_likelihood + rest_likelihood, preorder_mask_3
