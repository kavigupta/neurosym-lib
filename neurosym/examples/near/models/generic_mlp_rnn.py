from dataclasses import dataclass
from math import prod
from typing import Tuple

from torch import nn

from neurosym.examples.near.models.mlp import MLP, MLPConfig
from neurosym.examples.near.models.rnn import RNNConfig, Seq2ClassRNN, Seq2SeqRNN
from neurosym.examples.near.neural_hole_filler import NeuralHoleFiller
from neurosym.types.type import ArrowType, ListType, TensorType
from neurosym.types.type_annotated_object import TypeAnnotatedObject
from neurosym.types.type_shape import infer_output_shape
from neurosym.types.type_string_repr import render_type
from neurosym.types.type_with_environment import TypeWithEnvironment


@dataclass
class _MLPRNNInput:
    """
    Represents the input to an MLP/RNN module.
    """

    is_sequence: bool
    shape: Tuple[int]


def _classify_type(typ):
    """
    Classifies a type as either a tensor or a sequence of tensors. These are the only types supported by the MLP/RNN.
    """
    if isinstance(typ, ListType):
        assert isinstance(
            typ.element_type, TensorType
        ), f"Expected a list of tensors, but received {render_type(typ)}"
        # return "sequence", typ.element_type.shape
        return _MLPRNNInput(is_sequence=True, shape=typ.element_type.shape)
    if isinstance(typ, TensorType):
        # return "tensor", typ.shape
        return _MLPRNNInput(is_sequence=False, shape=typ.shape)
    return None


class GenericMLPRNNNeuralHoleFiller(NeuralHoleFiller):
    """
    Hole filler that attempts to use MLPs and RNNs to fill holes.

    :param hidden_size: The size of the hidden layer in the MLP/RNN.
    """

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def initialize_module(
        self, type_with_environment: TypeWithEnvironment
    ) -> nn.Module | None:

        typ = type_with_environment.typ
        input_types = []
        if isinstance(typ, ArrowType):
            input_types = list(typ.input_type)
            typ = typ.output_type

        input_types += [
            type_with_environment.env[x] for x in range(len(type_with_environment.env))
        ]

        input_classifications = [_classify_type(x) for x in input_types]
        output_classification = _classify_type(typ)
        if (
            any(x is None for x in input_classifications)
            or output_classification is None
        ):
            # If any of the types are not supported, return None.
            return None
        if output_classification.is_sequence and not any(
            x.is_sequence for x in input_classifications
        ):
            # If the output is a sequence, but none of the inputs are sequences, return None.
            return None
        return _GenericMLPRNNModule(
            hidden_size=self.hidden_size,
            input_types=input_types,
            output_type=typ,
            input_classifications=input_classifications,
            output_classification=output_classification,
        )


class _GenericMLPRNNModule(nn.Module):

    def __init__(
        self,
        hidden_size,
        input_types,
        output_type,
        input_classifications,
        output_classification,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.input_types = input_types
        self.output_type = output_type

        input_projections = []
        for inp in input_classifications:
            input_projections.append(nn.Linear(prod(inp.shape), self.hidden_size))
        self.input_projections = nn.ModuleList(input_projections)
        self.output_projection = nn.Linear(
            self.hidden_size, prod(output_classification.shape)
        )
        self.internal_module_type, self.internal_module = _internal_module(
            input_classifications, output_classification, self.hidden_size
        )

    def forward(self, *inputs, environment: Tuple[TypeAnnotatedObject, ...]):
        inputs = list(inputs) + [x.object_value for x in environment]
        type_shape, output_shape = infer_output_shape(
            self.input_types, [x.shape for x in inputs], self.output_type
        )
        inputs = [
            proj(type_shape.reshape_to_nlc(x))
            for proj, x in zip(self.input_projections, inputs)
        ]
        input_stacked = sum(inputs)
        if self.internal_module_type == "mlp":
            assert input_stacked.shape[1] == 1
            input_stacked = input_stacked.squeeze(1)
            output = self.internal_module(input_stacked, environment=environment)
        elif self.internal_module_type == "rnn":
            output = self.internal_module(input_stacked, environment=environment)
        else:
            raise ValueError(
                f"Unknown internal module type: {self.internal_module_type}"
            )
        output = self.output_projection(output)
        output = type_shape.unsquash_batch_axis(output)
        assert output.shape == output_shape
        return output


def _internal_module(input_classifications, output_classification, hidden_size):
    input_seq = any(
        input_classification.is_sequence
        for input_classification in input_classifications
    )
    if input_seq:
        config = RNNConfig(
            model_name="rnn",
            input_size=hidden_size,
            output_size=hidden_size,
            hidden_size=hidden_size,
        )
        return "rnn", (
            Seq2SeqRNN(config)
            if output_classification.is_sequence
            else Seq2ClassRNN(config)
        )

    config = MLPConfig(
        model_name="mlp",
        input_size=hidden_size,
        output_size=hidden_size,
        hidden_size=hidden_size,
    )
    return "mlp", MLP(config)
