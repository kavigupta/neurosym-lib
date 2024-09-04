from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import nn

from neurosym.examples.near.neural_dsl import compute_io_shape
from neurosym.types.type import Type

from .base import BaseConfig


@dataclass
class MLPConfig(BaseConfig):
    """
    Represents the configuration of an MLP module.

    :param input_size: The size of the input.
    :param hidden_size: The size of the hidden layer.
    :param output_size: The size of the output.
    :param dropout: Dropout rate to apply to the hidden layer.
    :param bias: Whether to use bias in the linear layers.
    :param nonlinearity: Nonlinearity to use in the hidden layer.
    :param loss: Loss function to use for training the model.
    """

    input_size: int
    hidden_size: int
    output_size: int
    dropout: float = 0.0
    bias: bool = True
    nonlinearity: str = "LeakyReLU"
    loss: str = "MSELoss"


class MLP(nn.Sequential):
    """
    Simple MLP module.

    :param config: Configuration for the MLP.
    """

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

    def forward(self, x, *, environment):
        # pylint: disable=arguments-differ,arguments-renamed
        del environment
        return super().forward(x)


def mlp_factory(
    hidden_size: int,
    dropout: float = 0.0,
    bias: bool = True,
    nonlinearity: str = "LeakyReLU",
    loss: str = "MSELoss",
):
    """
    Allows instantiating an MLP module with a given input and output size.

    :param hidden_size: Number of hidden units in the MLP.
    :param dropout: Dropout rate to apply to the hidden layer.
    :param bias: Whether to use bias in the linear layers.
    :param nonlinearity: Nonlinearity to use in the hidden layer.
    :param loss: Loss function to use for training the model.
    """

    def construct_model(typ: Type):
        input_shape, output_shape = compute_io_shape(typ)
        assert len(input_shape) == 1, "MLP takes a single input only."
        input_size = input_shape[0][-1]
        output_size = output_shape[-1]
        cfg = MLPConfig(
            model_name="mlp",
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            dropout=dropout,
            bias=bias,
            nonlinearity=nonlinearity,
            loss=loss,
        )
        return MLP(cfg)

    return construct_model
