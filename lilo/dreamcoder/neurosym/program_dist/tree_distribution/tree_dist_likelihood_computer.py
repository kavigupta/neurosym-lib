from types import NoneType
from typing import Callable, Tuple, Union

import numpy as np

from neurosym.program_dist.tree_distribution.preorder_mask.preorder_mask import (
    PreorderMask,
)
from neurosym.program_dist.tree_distribution.tree_distribution import TreeDistribution
from neurosym.programs.s_expression import SExpression


def compute_likelihood(
    tree_dist: TreeDistribution,
    program: SExpression,
    parents: Tuple[Tuple[int, int], ...],
    preorder_mask: PreorderMask,
    tracker: Union[NoneType, Callable[[SExpression, float], NoneType]],
):
    """
    Compute the likelihood of a program under a distribution.

    If ``tracker`` is not None, it will be called with each node and the likelihood
        of that node. This can be useful for debugging why a program has a certain
        likelihood.
    """
    start_position = parents[-1][1]
    top_symbol = tree_dist.symbol_to_index[program.symbol]
    likelihood = symbol_likelihood(
        tree_dist, parents, preorder_mask, start_position, top_symbol
    )
    if tracker is not None:
        tracker(program, likelihood)
    # only end early if the tracker is None,
    # because we want to call the tracker otherwise
    elif likelihood == -float("inf"):
        return -float("inf")
    preorder_mask.on_entry(start_position, top_symbol)
    order = tree_dist.ordering.order(top_symbol, len(program.children))
    for i, child in zip(order, [program.children[i] for i in order]):
        likelihood += compute_likelihood(
            tree_dist,
            child,
            (parents + ((top_symbol, i),))[-tree_dist.limit :],
            preorder_mask=preorder_mask,
            tracker=tracker,
        )
        if likelihood == -float("inf") and tracker is None:
            return -float("inf")
    preorder_mask.on_exit(start_position, top_symbol)
    return likelihood


def symbol_likelihood(tree_dist, parents, preorder_mask, start_position, top_symbol):
    if parents not in tree_dist.likelihood_arrays:
        return -float("inf")
    idx = tree_dist.index_within_distribution_list[parents].get(top_symbol, None)
    if idx is None:
        return -float("inf")
    syms, log_probs = tree_dist.likelihood_arrays[parents]
    mask = preorder_mask.compute_mask(start_position, syms)
    if not mask[idx]:
        return -float("inf")
    denominator = np.logaddexp.reduce(log_probs[mask])
    likelihood = log_probs[idx] - denominator

    assert likelihood <= 0, f"Likelihood is {likelihood}, expected <= 0"

    return likelihood
