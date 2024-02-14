from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn

from .base import BaseConfig


@dataclass
class ConstantConfig(BaseConfig):
    size: int
    init: str = "random"
    sample_categorical : bool = False

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
            self.dist = torch.distributions.Categorical(torch.ones(config.size) / config.size)

    def forward(self, x=None):
        dims = x.shape[:-1] + (self.config.size,)
        if self.config.sample_categorical:
            return self.dist.sample(dims[:-1])
        else:        
            return self.constant.expand(dims)


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
