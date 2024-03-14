from typing import Tuple
import neurosym as ns
import numpy as np

from neurosym.program_dist.tree_distribution.preorder_mask.preorder_mask import (
    PreorderMask,
)
from neurosym.program_dist.tree_distribution.tree_distribution import TreeDistribution


def compute_likelihood(
    tree_dist: TreeDistribution,
    program: ns.SExpression,
    parents: Tuple[Tuple[int, int], ...],
    preorder_mask: PreorderMask,
):
    """
    Compute the likelihood of a program under a distribution.
    """
    syms, log_probs = tree_dist.likelihood_arrays[parents]
    start_position = parents[-1][1]
    mask = preorder_mask.compute_mask(start_position, syms)
    denominator = np.logaddexp.reduce(log_probs[mask])
    top_symbol = tree_dist.symbol_to_index[program.symbol]
    likelihood = (
        tree_dist.distribution_dict[parents].get(top_symbol, -float("inf"))
        - denominator
    )
    if likelihood == -float("inf"):
        return -float("inf")
    preorder_mask.on_entry(start_position, top_symbol)
    for i, child in enumerate(program.children):
        likelihood += compute_likelihood(
            tree_dist,
            child,
            (parents + ((top_symbol, i),))[-tree_dist.limit :],
            preorder_mask=preorder_mask,
        )
    preorder_mask.on_exit(start_position, top_symbol)
    return likelihood
