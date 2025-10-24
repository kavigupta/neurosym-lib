from typing import Tuple

import numpy as np

from neurosym.program_dist.tree_distribution.preorder_mask.preorder_mask import (
    PreorderMask,
)
from neurosym.program_dist.tree_distribution.tree_distribution import TreeDistribution
from neurosym.programs.s_expression import SExpression


def attempt_to_sample_tree_dist(
    dist: TreeDistribution,
    rng: np.random.RandomState,
    depth_limit,
    ancestors: Tuple[Tuple[int, int], ...],
    parent: int,
    preorder_mask: PreorderMask,
) -> SExpression:
    """
    Attempt to sample a program from the distribution, conditioned on the depth limit.

    Args:
        rng: The random number generator to use.
        depth_limit: The maximum depth of the program.
        parents: The parents of the current node, to a limit of dist.limit

    Raises:
        TooDeepError: If the program is too deep.
    """

    if depth_limit < 0:
        raise TooDeepError()
    root_sym, root_arity = dist.symbols[parent]
    children = [None] * root_arity
    for i in dist.ordering.order(parent, root_arity):
        key = ancestors + ((parent, i),)
        key = key[-dist.limit :]
        possibilites, weights = dist.sampling_dict_arrays[key]
        mask = preorder_mask.compute_mask(i, possibilites)
        possibilites, weights = possibilites[mask], weights[mask]
        if len(possibilites) == 0:
            raise ValueError(f"No valid productions for {key}")
        weights /= weights.sum()
        child_idx = rng.choice(possibilites, p=weights)
        preorder_mask.on_entry(i, child_idx)
        children[i] = attempt_to_sample_tree_dist(
            dist,
            rng,
            depth_limit - 1,
            ancestors=key,
            parent=child_idx,
            preorder_mask=preorder_mask,
        )
        preorder_mask.on_exit(i, child_idx)
    return SExpression(root_sym, tuple(children))


def sample_tree_dist(
    dist: TreeDistribution, rng: np.random.RandomState, depth_limit
) -> SExpression:
    """
    Sample a program from the distribution, conditioned on the depth limit.

    Args:
        rng: The random number generator to use.
        depth_limit: The maximum depth of the program.
        parents: The parents of the current node, to a limit of dist.limit
    """
    while True:
        try:
            preorder_mask = dist.mask_constructor(dist)
            preorder_mask.on_entry(0, 0)
            return attempt_to_sample_tree_dist(
                dist,
                rng,
                depth_limit,
                ancestors=(),
                parent=0,
                preorder_mask=preorder_mask,
            )
        except TooDeepError:
            continue


class TooDeepError(Exception):
    pass
