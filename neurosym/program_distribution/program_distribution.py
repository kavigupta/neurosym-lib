from abc import ABC, abstractmethod
from ast import Tuple
from typing import List

import numpy as np
import torch

from neurosym.programs.s_expression import SExpression

from ..dsl.dsl import DSL


class ProgramDistribution(ABC):
    """
    Represents a distribution over programs.

    Important concepts:
        * tensor: a neural-network friendly representation of the distribution's parameters.
            E.g., might be logits and exclude sites that must be 0 or 1.
        * parameter: a sampler-friendly representation of the distribution's parameters.
    """

    @abstractmethod
    def tensor_shape(self, dsl: DSL) -> Tuple[int, ...]:
        """
        Return the number of parameters in the tensor representation of the distribution.
        """
        pass

    @abstractmethod
    def tensor_to_parameters(self, dsl: DSL, tensor: torch.tensor) -> np.ndarray:
        """
        Return the parameter representation of the PCFG tensor, starting with a tensor.
        """
        pass

    @abstractmethod
    def tensor_loss(
        self, dsl: DSL, tensor: torch.Tensor, programs: List[SExpression]
    ) -> torch.Tensor:
        """
        Return the loss of the given tensor, as translated to the praameters.

        E.g., this could be a log-likelihood loss.
        """
        pass

    @abstractmethod
    def sample(
        self, dsl: DSL, parameters: np.ndarray, rng: np.random.RandomState
    ) -> SExpression:
        """
        Sample a program from the PCFG with the given parameters.
        """
        pass
