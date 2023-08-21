from dataclasses import dataclass
import torch
from torch import nn
from collections import OrderedDict
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
