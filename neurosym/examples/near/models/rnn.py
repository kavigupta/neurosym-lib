from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn

from .base import BaseConfig


@dataclass
class RNNConfig(BaseConfig):
    input_size: int
    hidden_size: int
    output_size: int


class RNN(nn.Module):
    """Abstract RNN module."""

    def __init__(self, config: RNNConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.rnn = nn.RNN(
            config.input_size,
            config.hidden_size,
            batch_first=True,
            num_layers=1,
            bidirectional=False,
            dropout=0.0,
            nonlinearity="tanh",
            bias=True,
        )
        self.fc = nn.Linear(config.hidden_size, config.output_size)

    def seq2class(self, x, hidden: torch.Tensor = None):
        """
        :param x : (batch_size, seq_length, input_size)
        :return out : (batch_size, output_size)
        """
        b, _, _ = x.shape
        h0 = (
            torch.zeros(1, b, self.hidden_size, device=x.device)
            if hidden is None
            else hidden
        )
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

    def seq2seq(self, x, hidden: torch.Tensor = None):
        """
        :param x : (batch_size, seq_length, input_size)
        :return out : (batch_size, seq_length, output_size)
        """
        b, s, _ = x.shape
        h0 = (
            torch.zeros(1, b, self.hidden_size, device=x.device)
            if hidden is None
            else hidden
        )
        out, _ = self.rnn(x, h0)
        out = self.fc(out.contiguous().view(b * s, -1)).view(b, s, -1)
        return out

    def forward(self, inp, hidden: torch.Tensor = None):
        pass


class Seq2SeqRNN(RNN):
    def forward(self, inp: torch.Tensor, hidden: torch.Tensor = None):
        return self.seq2seq(inp, hidden)


class Seq2ClassRNN(RNN):
    def forward(self, inp: torch.Tensor, hidden: torch.Tensor = None):
        return self.seq2class(inp, hidden)


def rnn_factory_seq2seq(**kwargs):
    """
    Allows instantiating an MLP module with a given input and output size.
    """

    def construct_model(input_shape: List[Tuple[int]], output_shape: Tuple[int]):
        assert len(input_shape) == 1, "MLP takes a single input only."
        input_size = input_shape[0][-1]
        output_size = output_shape[-1]
        cfg = RNNConfig(
            model_name="rnn",
            input_size=input_size,
            output_size=output_size,
            **kwargs,
        )
        return Seq2SeqRNN(cfg)

    return construct_model


def rnn_factory_seq2class(**kwargs):
    """
    Allows instantiating an MLP module with a given input and output size.
    """

    def construct_model(input_shape: List[Tuple[int]], output_shape: Tuple[int]):
        assert len(input_shape) == 1, "MLP takes a single input only."
        input_size = input_shape[0][-1]
        output_size = output_shape[-1]
        cfg = RNNConfig(
            model_name="rnn",
            input_size=input_size,
            output_size=output_size,
            **kwargs,
        )
        return Seq2ClassRNN(cfg)

    return construct_model
