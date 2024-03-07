from abc import ABC, abstractmethod
from typing import List, TypeVar

import numpy as np
import torch

from neurosym.dsl.dsl import DSL
from neurosym.program_dist.tree_dist_enumerator import (
    DEFAULT_CHUNK_SIZE,
    TreeDistribution,
    enumerate_tree_dist,
)
from neurosym.programs.s_expression import SExpression

ProgramDistribution = TypeVar("ProgramDistribution")
ProgramsCountTensor = TypeVar("ProgramsCountTensor")


class ProgramDistributionFamily(ABC):
    """
    Abstract class representing a family of distributions over programs.
    """

    @abstractmethod
    def underlying_dsl(self) -> DSL:
        """
        Returns the DSL that this distribution is over.
        """

    @abstractmethod
    def parameters_shape(self) -> List[int]:
        """
        Returns the shape of the parameters of this distribution.
        """

    @abstractmethod
    def with_parameters(self, parameters: torch.Tensor) -> List[ProgramDistribution]:
        """
        Initializes a distribution from the given parameters. The parameters
            should have the shape (batch_size, *self.parameters_shape()).
        """

    @abstractmethod
    def count_programs(self, data: List[List[SExpression]]) -> ProgramsCountTensor:
        """
        For each program, count its components' occurrences in the data. This
            depends on the type of distribution.
        """

    @abstractmethod
    def counts_to_distribution(
        self, counts: ProgramsCountTensor
    ) -> ProgramDistribution:
        """
        Converts the counts to a distribution.
        """

    @abstractmethod
    def parameter_difference_loss(
        self, parameters: torch.tensor, actual: ProgramsCountTensor
    ) -> torch.float32:
        """
        Returns the loss between the parameters and actual counts, for
            several count tensors. Keep in mind that the parameters
            here should have a batch dimension, and
            parameters.shape[0] == len(actual).
        """

    @abstractmethod
    def sample(
        self,
        dist: ProgramDistribution,
        num_samples: int,
        rng: np.random.RandomState,
        *,
        depth_limit=float("inf"),
    ) -> SExpression:
        """
        Samples programs from this distribution.
        """

    @abstractmethod
    def enumerate(
        self,
        dist: ProgramDistribution,
        *,
        min_likelihood: float = float("-inf"),
        chunk_size: float = DEFAULT_CHUNK_SIZE,
    ):
        """
        Enumerate all programs using iterative deepening. Yields (program, likelihood).

        Args:
            dist: The distribution to sample from.
            chunk_size: The amount of likelihood to consider at once. If this is
                too small, we will spend a lot of time doing the same work over and
                over again. If this is too large, we will spend a lot of time
                doing work that we don't need to do.
        """


class TreeProgramDistributionFamily(ProgramDistributionFamily):
    """
    See `tree_dist_enumerator.py` for more information.
    """

    @abstractmethod
    def compute_tree_distribution(
        self, distribution: ProgramDistribution
    ) -> TreeDistribution:
        """
        Returns a tree distribution representing the given program distribution.
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

    def enumerate(
        self,
        dist: ProgramDistribution,
        *,
        min_likelihood: float = float("-inf"),
        chunk_size: float = DEFAULT_CHUNK_SIZE,
    ):
        tree_dist = self.tree_distribution(dist)
        return enumerate_tree_dist(
            tree_dist, min_likelihood=min_likelihood, chunk_size=chunk_size
        )

    def compute_likelihood(
        self, dist: ProgramDistribution, program: SExpression
    ) -> float:
        """
        Compute the likelihood of a program under a distribution.
        """
        return self.tree_distribution(dist).compute_likelihood(program)