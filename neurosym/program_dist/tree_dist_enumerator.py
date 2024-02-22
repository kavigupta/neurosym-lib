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
from dataclasses import dataclass
from typing import Dict, List, Tuple

from neurosym.programs.s_expression import SExpression

# Let c be the chunk size, h be the likelihood of the program we are trying to
# find, and T be the number of iterations we need to do.
#
# We have that the total time taken is
#     sum_{t=0}^{T - 1} e^{c * t} + f * e^{c * T}
# where f is the fraction of the last chunk. We can compute this in expectation
# as 1/2, which leads us to
#     sum_{t=0}^{T - 1} e^{c * t} + e^{c * T} / 2
# which can be computed as
#     (e^{c * (T + 1)} - 1) / (e^c - 1) - e^{c * T} / 2
# which can be approximated closely as
#     e^{cT} * e^c / (e^c - 1) - e^{c * T} / 2
# which is equal to
#     e^{cT} * (e^c / (e^c - 1) - 1/2)
# we then know that T = ceil(h / c), and letting g = T - h / c, we have
#     e^{cT} = e^{c * (h / c + g)} = e^h * e^{c * g}
# if we assume g is uniformly distributed between 0 and 1, we have
#     e^{c * g} = (e^c - 1) / c
# we then have that the expected time is
#     (e^c - 1) / c * (e^c / (e^c - 1) - 1/2)
# which is optimized per wolfram at
#     W(1/e) + 1
# which is approximately 1.28.
DEFAULT_CHUNK_SIZE = 1.278


@dataclass
class TreeDistribution:
    """
    Distribution over SExpressions as trees.

    Internally, we represent the productions in the language as integers, which we
        call indices.
    """

    limit: int
    # input: tuple of ancestor production indices followed by
    #   the position of the node in its parent's children
    # output: list of (production index, likelihood) pairs
    distribution: Dict[Tuple[int, ...], List[Tuple[int, float]]]
    # production index -> (symbol, arity). at 0 should be the root.
    symbols: List[Tuple[str, int]]


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
            tree_dist, likelihood_bound, (0,), 0
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
    parents: Tuple[int],
    position: int,
):
    """
    Enumerate all programs that are within the likelihood range, with the given parents.
    """

    if min_likelihood > 0:
        # We can stop searching deeper.
        return

    assert len(parents) <= tree_dist.limit

    # Performed recursively for now.

    distribution = tree_dist.distribution[(*parents, position)]
    for node, likelihood in distribution:
        new_parents = parents + (node,)
        new_parents = new_parents[-tree_dist.limit :]
        symbol, arity = tree_dist.symbols[node]
        for children, child_likelihood in enumerate_children_and_likelihoods_dfs(
            tree_dist,
            min_likelihood - likelihood,
            new_parents,
            num_children=arity,
        ):
            yield SExpression(symbol, children), child_likelihood + likelihood


def enumerate_children_and_likelihoods_dfs(
    tree_dist: TreeDistribution,
    min_likelihood: float,
    parents: Tuple[int],
    num_children: int,
):
    """
    Enumerate all children and their likelihoods.
    """

    if num_children == 0:
        yield [], 0
        return

    for last_child, last_likelihood in enumerate_tree_dist_dfs(
        tree_dist, min_likelihood, parents, num_children - 1
    ):
        for rest_children, rest_likelihood in enumerate_children_and_likelihoods_dfs(
            tree_dist, min_likelihood - last_likelihood, parents, num_children - 1
        ):
            yield rest_children + [last_child], last_likelihood + rest_likelihood
