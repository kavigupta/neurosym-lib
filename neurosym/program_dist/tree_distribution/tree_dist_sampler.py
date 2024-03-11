import numpy as np

from neurosym.program_dist.tree_distribution.tree_distribution import TreeDistribution
from neurosym.programs.s_expression import SExpression


def attempt_to_sample_tree_dist(
    dist: TreeDistribution,
    rng: np.random.RandomState,
    depth_limit,
    parents: list[int],
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
        child_idx = rng.choice(possibilites, p=weights)
        child_parents = parents + (child_idx,)
        child_parents = child_parents[-dist.limit :]
        children.append(
            attempt_to_sample_tree_dist(
                dist, rng, depth_limit - 1, parents=child_parents
            )
        )
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
            return attempt_to_sample_tree_dist(dist, rng, depth_limit, parents)
        except TooDeepError:
            continue


class TooDeepError(Exception):
    pass
