from abc import ABC, abstractmethod
from typing import List, TypeVar

import numpy as np
import torch

from neurosym.dsl.dsl import DSL
from neurosym.program_dist.enumeration_chunk_size import DEFAULT_CHUNK_SIZE
from neurosym.programs.s_expression import SExpression

ProgramDistribution = TypeVar("ProgramDistribution")
ProgramDistributionBatch = TypeVar("ProgramDistributionBatch")
ProgramCountsTensorBatch = TypeVar("ProgramCountsTensorBatch")


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
    def with_parameters(self, parameters: torch.Tensor) -> ProgramDistributionBatch:
        """
        Initializes a distribution from the given parameters. The parameters
            should have the shape (batch_size, *self.parameters_shape()).
        """

    @abstractmethod
    def count_programs(self, data: List[List[SExpression]]) -> ProgramCountsTensorBatch:
        """
        For each program, count its components' occurrences in the data. This
            depends on the type of distribution.
        """

    @abstractmethod
    def counts_to_distribution(
        self, counts: ProgramCountsTensorBatch
    ) -> ProgramDistributionBatch:
        """
        Converts the counts to a distribution.
        """

    @abstractmethod
    def parameter_difference_loss(
        self, parameters: torch.tensor, actual: ProgramCountsTensorBatch
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
