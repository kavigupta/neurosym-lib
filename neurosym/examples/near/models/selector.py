from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn

from .base import BaseConfig


@dataclass
class SelectorConfig(BaseConfig):
    input_size: int
    output_size: int


class Selector(nn.Module):
    """Simple Selector module."""

    def __init__(self, config: SelectorConfig):
        super().__init__()
        self.config = config
        self.ln = nn.Linear(config.input_size, config.output_size)

    def forward(self, x=None):
        logits = self.ln(x)
        masked_logits = torch.nn.functional.gumbel_softmax(logits, tau=1, hard=True)
        return masked_logits

def selector_factory(**kwargs):
    """
    Allows instantiating an selector module with a given input and output size.
    """
    input_dim = kwargs['input_dim']
    kwargs.pop('input_dim')

    def construct_model(input_shape: List[Tuple[int]], output_shape: Tuple[int]):
        assert len(input_shape) == 0
        cfg = SelectorConfig(
            model_name="Selector",
            input_size=input_dim,
            output_size=output_shape[-1],
            **kwargs,
        )
        return Selector(cfg)

    return construct_model
