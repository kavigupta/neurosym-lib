from abc import ABC, abstractmethod
from typing import List, TypeVar

import numpy as np
import torch

from neurosym.dsl.dsl import DSL
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
        depth_limit=float("inf")
    ) -> SExpression:
        """
        Samples programs from this distribution.
        """
