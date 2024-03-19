from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from types import NoneType
from typing import Callable, Dict, List, Tuple, Union

import numpy as np

from neurosym.program_dist.distribution import (
    ProgramDistribution,
    ProgramDistributionFamily,
)
from neurosym.program_dist.enumeration_chunk_size import DEFAULT_CHUNK_SIZE
from neurosym.program_dist.tree_distribution.ordering import NodeOrdering
from neurosym.program_dist.tree_distribution.preorder_mask.preorder_mask import (
    PreorderMask,
)
from neurosym.programs.s_expression import SExpression


@dataclass
class TreeDistribution:
    """
    Distribution over SExpressions as trees.

    Internally, we represent the productions in the language as integers, which we
        call indices.
    """

    limit: int
    # input: tuple of tuples of (ancestor index, position)
    #        which is the path to the current node, with the
    #        most immediate ancestor at the end.
    # output: list of (production index, likelihood) pairs
    distribution: Dict[Tuple[Tuple[int, int], ...], List[Tuple[int, float]]]
    # production index -> (symbol, arity). at 0 should be the root.
    symbols: List[Tuple[str, int]]
    # Preorder mask constructor
    mask_constructor: Callable[["TreeDistribution"], PreorderMask]
    # Node ordering
    node_ordering: Callable[["TreeDistribution"], NodeOrdering]

    @cached_property
    def symbol_to_index(self) -> Dict[str, int]:
        return {symbol: i for i, (symbol, _) in enumerate(self.symbols)}

    @cached_property
    def index_within_distribution_list(
        self,
    ) -> Dict[Tuple[Tuple[int, int], ...], Dict[int, int]]:
        return {
            k: {x: i for i, (x, _) in enumerate(v)}
            for k, v in self.distribution.items()
        }

    @cached_property
    def distribution_dict(self) -> Dict[Tuple[Tuple[int, int], ...], Dict[int, float]]:
        return {k: dict(v) for k, v in self.distribution.items()}

    @cached_property
    def likelihood_arrays(
        self,
    ) -> Dict[Tuple[Tuple[int, int], ...], Tuple[np.ndarray, np.ndarray]]:
        return {
            k: (
                np.array([x[0] for x in v]),
                np.array([x[1] for x in v]),
            )
            for k, v in self.distribution.items()
        }

    @cached_property
    def sampling_dict_arrays(
        self,
    ) -> Dict[Tuple[Tuple[int, int], ...], Tuple[np.ndarray, np.ndarray]]:
        return {
            k: (syms, np.exp(log_probs))
            for k, (syms, log_probs) in self.likelihood_arrays.items()
        }

    @cached_property
    def ordering(self) -> NodeOrdering:
        return self.node_ordering(self)


class TreeProgramDistributionFamily(ProgramDistributionFamily):
    """
    See `tree_dist_enumerator.py` for more information.
    """

    @abstractmethod
    def compute_tree_distribution(
        self, distribution: Union[ProgramDistribution, NoneType]
    ) -> TreeDistribution:
        """
        Returns a tree distribution representing the given program distribution.

        If `distribution` is `None`, returns a tree distribution with all fields
            initialized except `distribution`. This is useful for tasks where you want
            the skeleton of the tree distribution, but don't need the actual distribution.
        """

    def tree_distribution(self, distribution: ProgramDistribution) -> TreeDistribution:
        """
        Cached version of `compute_tree_distribution`.
        """
        # This is a bit of a hack, but it reduces the need to pass around
        # the tree distribution everywhere, or to compute it multiple times.
        # pylint: disable=protected-access
        if not hasattr(distribution, "_tree_distribution"):
            distribution._tree_distribution = self.compute_tree_distribution(
                distribution
            )
        return distribution._tree_distribution

    @cached_property
    def tree_distribution_skeleton(self) -> TreeDistribution:
        """
        Cached version of `compute_tree_distribution(None)`.
        """

        return self.compute_tree_distribution(None)

    def enumerate(
        self,
        dist: ProgramDistribution,
        *,
        min_likelihood: float = float("-inf"),
        chunk_size: float = DEFAULT_CHUNK_SIZE,
    ):
        # pylint: disable=cyclic-import
        from neurosym.program_dist.tree_distribution.tree_dist_enumerator import (
            enumerate_tree_dist,
        )

        tree_dist = self.tree_distribution(dist)

        return enumerate_tree_dist(
            tree_dist, min_likelihood=min_likelihood, chunk_size=chunk_size
        )

    def compute_likelihood(
        self,
        dist: ProgramDistribution,
        program: SExpression,
        tracker: Union[NoneType, Callable[[SExpression, float], NoneType]] = None,
    ) -> float:
        """
        Compute the likelihood of a program under a distribution.
        """
        # pylint: disable=cyclic-import
        from .tree_dist_likelihood_computer import compute_likelihood

        dist = self.tree_distribution(dist)
        preorder_mask = dist.mask_constructor(dist)
        preorder_mask.on_entry(0, 0)
        return compute_likelihood(dist, program, ((0, 0),), preorder_mask, tracker)

    def compute_likelihood_per_node(
        self,
        dist: ProgramDistribution,
        program: SExpression,
    ) -> Dict[SExpression, float]:
        """
        Compute the likelihood of a program under a distribution.
        """
        likelihoods = []

        self.compute_likelihood(
            dist,
            program,
            lambda nodes, likelihood: likelihoods.append((nodes, likelihood)),
        )
        return likelihoods

    def sample(
        self,
        dist: ProgramDistribution,
        rng: np.random.RandomState,
        *,
        depth_limit=float("inf"),
    ) -> SExpression:
        # pylint: disable=cyclic-import

        from neurosym.program_dist.tree_distribution.tree_dist_sampler import (
            sample_tree_dist,
        )

        tree_dist = self.tree_distribution(dist)
        element = sample_tree_dist(tree_dist, rng, depth_limit=depth_limit)
        assert element.symbol == "<root>"
        [element] = element.children
        return element
