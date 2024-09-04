from dataclasses import dataclass

import torch
from torch import nn

from neurosym.examples.near.neural_dsl import compute_io_shape
from neurosym.types.type import Type

from .base import BaseConfig


@dataclass
class RNNConfig(BaseConfig):
    """
    Represents the configuration of an RNN module.

    :param input_size: The size of the input.
    :param hidden_size: The size of the hidden layer.
    :param output_size: The size of the output.
    """

    input_size: int
    hidden_size: int
    output_size: int


class _RNN(nn.Module):
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

    def _seq2class(self, x, hidden: torch.Tensor = None):
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

    def _seq2seq(self, x, hidden: torch.Tensor = None):
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

    def forward(self, inp, hidden: torch.Tensor = None, *, environment):
        pass


class Seq2SeqRNN(_RNN):
    """
    RNN module for sequence-to-sequence tasks.

    :param config: Configuration for the RNN.
    """

    def forward(self, inp: torch.Tensor, hidden: torch.Tensor = None, *, environment):
        return self._seq2seq(inp, hidden)


class Seq2ClassRNN(_RNN):
    """
    RNN module for sequence-to-class tasks.

    :param config: Configuration for the RNN.
    """

    def forward(self, inp: torch.Tensor, hidden: torch.Tensor = None, *, environment):
        return self._seq2class(inp, hidden)


def rnn_factory_seq2seq(hidden_size: int):
    """
    Allows instantiating a RNN module for sequence-to-sequence tasks, with a given hidden size.

    :param hidden_size: Size of the hidden layer in the RNN.
    """

    def construct_model(typ: Type):
        input_shape, output_shape = compute_io_shape(typ)
        assert len(input_shape) == 1, "RNN takes a single input only."
        input_size = input_shape[0][-1]
        output_size = output_shape[-1]
        cfg = RNNConfig(
            model_name="rnn",
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
        )
        return Seq2SeqRNN(cfg)

    return construct_model


def rnn_factory_seq2class(hidden_size: int):
    """
    Allows instantiating a RNN module for sequence-to-class tasks, with a given hidden size.

    :param hidden_size: Size of the hidden layer in the RNN.
    """

    def construct_model(typ: Type):
        input_shape, output_shape = compute_io_shape(typ)
        assert len(input_shape) == 1, "RNN takes a single input only."
        input_size = input_shape[0][-1]
        output_size = output_shape[-1]
        cfg = RNNConfig(
            model_name="rnn",
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
        )
        return Seq2ClassRNN(cfg)

    return construct_model
