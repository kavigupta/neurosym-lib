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

import bisect
import itertools
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from torch import NoneType

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
    use_cache=True,
):
    """
    Enumerate all programs using iterative deepening.

    :param tree_dist: The distribution to sample from.
    :param min_likelihood: The minimum likelihood to consider.
    :param chunk_size: The amount of likelihood to consider at once. If this is
        too small, we will spend a lot of time doing the same work over and
        over again. If this is too large, we will spend a lot of time
        doing work that we don't need to do.
    :param use_cache: Whether to use a cache to store intermediate results.
        This will be disabled if the mask does not support caching.
    """
    for chunk in itertools.count(1):
        likelihood_bound = -chunk * chunk_size
        preorder_mask = tree_dist.mask_constructor(tree_dist)
        cache = {} if use_cache and preorder_mask.can_cache else None
        preorder_mask.on_entry(0, 0)
        skipped_something = _HasSkippedSomething()
        for program, likelihood in _enumerate_tree_dist_dfs(
            tree_dist,
            likelihood_bound,
            ((0, 0),),
            preorder_mask,
            cache,
            skipped_something,
        ):
            if (
                max(likelihood_bound, min_likelihood)
                < likelihood
                <= likelihood_bound + chunk_size
            ):
                yield program, likelihood
        if not skipped_something.skipped_something:
            return
        if likelihood_bound <= min_likelihood:
            return


def _remove_below_threshold(
    results: List[Tuple[SExpression, float]], min_likelihood: float
):
    """
    Remove all results below the threshold.
    """
    # binary search
    index = bisect.bisect_left(results, min_likelihood, key=lambda x: x[1])
    return results[index:]


class _HasSkippedSomething:
    """
    Tracks whether we have skipped something.
    """

    def __init__(self):
        self._skipped_something = False

    def notify_skip(self):
        self._skipped_something = True

    @property
    def skipped_something(self):
        return self._skipped_something


def _enumerate_tree_dist_dfs(
    tree_dist: TreeDistribution,
    min_likelihood: float,
    parents: Tuple[Tuple[int, int], ...],
    preorder_mask: PreorderMask,
    cache: Union[NoneType, Dict[Any, List[Tuple[SExpression, float]]]],
    skipped_something: _HasSkippedSomething,
):
    if cache is not None:
        key = preorder_mask.cache_key(parents), parents
        if key in cache:
            old_results, old_min_likelihood, have_skipped_something = cache[key]
            if old_min_likelihood <= min_likelihood:
                if have_skipped_something:
                    skipped_something.notify_skip()
                return _remove_below_threshold(old_results, min_likelihood)

    if cache is not None:
        skipped_something_child = _HasSkippedSomething()
    else:
        skipped_something_child = skipped_something

    generator = _enumerate_tree_dist_dfs_uncached(
        tree_dist,
        min_likelihood,
        parents,
        preorder_mask,
        cache,
        skipped_something_child,
    )
    if cache is not None:
        generator = sorted(generator, key=lambda x: x[1])
        cache[key] = (
            generator,
            min_likelihood,
            skipped_something_child.skipped_something,
        )
        if skipped_something_child.skipped_something:
            skipped_something.notify_skip()
    return generator


def _enumerate_tree_dist_dfs_uncached(
    tree_dist: TreeDistribution,
    min_likelihood: float,
    parents: Tuple[Tuple[int, int], ...],
    preorder_mask: PreorderMask,
    cache: Union[NoneType, Dict[Any, List[Tuple[SExpression, float]]]],
    skipped_something: _HasSkippedSomething,
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
        likelihood = np.float64(likelihood)
        new_parents = parents + (node,)
        new_parents = new_parents[-tree_dist.limit :]
        symbol, arity = tree_dist.symbols[node]
        undo_entry = preorder_mask.on_entry(position, node)
        for children, child_likelihood in _enumerate_children_and_likelihoods_dfs(
            tree_dist,
            min_likelihood - likelihood,
            parents,
            node,
            num_children=arity,
            starting_index=0,
            order=tree_dist.ordering.order(node, arity),
            preorder_mask=preorder_mask,
            cache=cache,
            skipped_something=skipped_something,
        ):
            if child_likelihood + likelihood < min_likelihood:
                skipped_something.notify_skip()
                continue
            undo_exit = preorder_mask.on_exit(position, node)
            yield SExpression(
                symbol, [children[i] for i in range(arity)]
            ), child_likelihood + likelihood
            undo_exit()
        undo_entry()


def _enumerate_children_and_likelihoods_dfs(
    tree_dist: TreeDistribution,
    min_likelihood: float,
    parents: Tuple[Tuple[int, int], ...],
    most_recent_parent: int,
    num_children: int,
    starting_index: int,
    order: List[int],
    preorder_mask: PreorderMask,
    cache: Union[NoneType, Dict[Any, List[Tuple[SExpression, float]]]],
    skipped_something: _HasSkippedSomething,
):
    """
    Enumerate all children and their likelihoods.
    """

    if min_likelihood > 0:
        # We can stop searching deeper.
        skipped_something.notify_skip()
        return

    if starting_index == num_children:
        yield {}, 0
        return
    new_parents = parents + ((most_recent_parent, order[starting_index]),)
    new_parents = new_parents[-tree_dist.limit :]

    for first_child, first_likelihood in _enumerate_tree_dist_dfs(
        tree_dist, min_likelihood, new_parents, preorder_mask, cache, skipped_something
    ):
        for (
            rest_children,
            rest_likelihood,
        ) in _enumerate_children_and_likelihoods_dfs(
            tree_dist,
            min_likelihood - first_likelihood,
            parents,
            most_recent_parent,
            num_children,
            starting_index + 1,
            order,
            preorder_mask,
            cache=cache,
            skipped_something=skipped_something,
        ):
            rest_children[order[starting_index]] = first_child
            yield rest_children, first_likelihood + rest_likelihood
