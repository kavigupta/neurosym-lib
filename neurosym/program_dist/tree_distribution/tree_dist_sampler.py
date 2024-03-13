from typing import Tuple

import numpy as np

from neurosym.program_dist.tree_distribution.tree_distribution import TreeDistribution
from neurosym.programs.s_expression import SExpression


def attempt_to_sample_tree_dist(
    dist: TreeDistribution,
    rng: np.random.RandomState,
    depth_limit,
    ancestors: Tuple[Tuple[int, int], ...],
    parent: int,
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
    children = []
    for i in range(root_arity):
        key = ancestors + ((parent, i),)
        key = key[-dist.limit :]
        possibilites, weights = dist.sampling_dict_arrays[key]
        child_idx = rng.choice(possibilites, p=weights)
        children.append(
            attempt_to_sample_tree_dist(
                dist, rng, depth_limit - 1, ancestors=key, parent=child_idx
            )
        )
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
            return attempt_to_sample_tree_dist(
                dist, rng, depth_limit, ancestors=(), parent=0
            )
        except TooDeepError:
            continue


class TooDeepError(Exception):
    pass
