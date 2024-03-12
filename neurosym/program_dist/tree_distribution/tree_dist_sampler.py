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
    parents: list[int],
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
    root_sym, root_arity = dist.symbols[parents[-1]]
    children = []
    for i in range(root_arity):
        key = parents + (i,)
        possibilites, weights = dist.sampling_dict_arrays[key]
        mask = preorder_mask.compute_mask(i, possibilites)
        possibilites, weights = possibilites[mask], weights[mask]
        if len(possibilites) == 0:
            raise ValueError(f"No valid productions for {key}")
        weights /= weights.sum()
        child_idx = rng.choice(possibilites, p=weights)
        preorder_mask.on_entry(i, child_idx)
        child_parents = parents + (child_idx,)
        child_parents = child_parents[-dist.limit :]
        children.append(
            attempt_to_sample_tree_dist(
                dist,
                rng,
                depth_limit - 1,
                parents=child_parents,
                preorder_mask=preorder_mask,
            )
        )
        preorder_mask.on_exit(i, child_idx)
    return SExpression(root_sym, tuple(children))


def sample_tree_dist(
    dist: TreeDistribution,
    rng: np.random.RandomState,
    depth_limit,
    parents: list[int] = (0,),
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
                dist, rng, depth_limit, parents, preorder_mask
            )
        except TooDeepError:
            continue


class TooDeepError(Exception):
    pass
