from dataclasses import dataclass
from math import prod
from typing import List, Tuple

import torch
from torch import nn

from .base import BaseConfig


@dataclass
class ConstantConfig(BaseConfig):
    size: int
    init: str = "random"
    sample_categorical: bool = False


class Constant(nn.Module):
    """Simple Constant module."""

    def __init__(self, config: ConstantConfig):
        super().__init__()
        self.config = config
        match config.init:
            case "random":
                self.constant = torch.nn.Parameter(torch.randn(config.size))
            case "zeros":
                self.constant = torch.nn.Parameter(torch.zeros(config.size))

        if config.sample_categorical:
            self.probs = torch.nn.Parameter(torch.ones(config.size) / config.size)

    def forward(self, x=None, environment=()):
        """Forward pass for Constant module.

        :param x: Optional input tensor (only used for shape inference)
        :param environment: Environment parameter (required by new API, not used)
        """
        del environment  # Not used
        dims = x.shape[:-1] + (1,) if x is not None else (1,)
        if self.config.sample_categorical:
            out = torch.multinomial(
                self.probs, num_samples=prod(dims), replacement=True
            ).reshape(dims)
        else:
            out = self.constant.unsqueeze(-1).expand(dims)
        return out


def constant_factory(**kwargs):
    """
    Allows instantiating an constant module with a given input and output size.
    """

    def construct_model(input_shape: List[Tuple[int]], output_shape: Tuple[int]):
        assert len(input_shape) == 0, "Constant takes no input."
        size = output_shape[-1]
        cfg = ConstantConfig(
            model_name="Constant",
            size=size,
            **kwargs,
        )
        return Constant(cfg)

    return construct_model
