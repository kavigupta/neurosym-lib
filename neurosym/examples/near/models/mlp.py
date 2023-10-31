from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn

from .base import BaseConfig


@dataclass
class MLPConfig(BaseConfig):
    input_size: int
    hidden_size: int
    output_size: int
    dropout: float = 0.0
    bias: bool = True
    nonlinearity: str = "LeakyReLU"
    loss: str = "MSELoss"


class MLP(nn.Sequential):
    """Simple MLP module."""

    def __init__(self, config: MLPConfig):
        self.config = config
        nonlinearity = getattr(nn, config.nonlinearity)
        model_dict = OrderedDict(
            fc1=nn.Linear(config.input_size, config.hidden_size, bias=config.bias),
            act1=nonlinearity(),
            dp1=nn.Dropout(p=config.dropout),
            fc3=nn.Linear(config.hidden_size, config.output_size, bias=config.bias),
        )
        super().__init__(model_dict)

        @torch.no_grad()
        def init_weights(m):
            if isinstance(m, nn.Linear):
                m.weight = nn.init.xavier_normal_(
                    m.weight, gain=torch.nn.init.calculate_gain("leaky_relu")
                )

        self.apply(init_weights)


def mlp_factory(**kwargs):
    """
    Allows instantiating an MLP module with a given input and output size.
    """

    def construct_model(input_shape: List[Tuple[int]], output_shape: Tuple[int]):
        assert len(input_shape) == 1, "MLP takes a single input only."
        input_size = input_shape[0][-1]
        output_size = output_shape[-1]
        cfg = MLPConfig(
            model_name="mlp",
            input_size=input_size,
            output_size=output_size,
            **kwargs,
        )
        return MLP(cfg)

    return construct_model
