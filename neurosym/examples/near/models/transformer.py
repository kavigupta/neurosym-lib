import math
from typing import Tuple

import numpy as np
import torch
from torch import nn

from neurosym.examples.near.neural_hole_filler import NeuralHoleFiller
from neurosym.types.type import ArrowType, Type
from neurosym.types.type_annotated_object import TypeAnnotatedObject
from neurosym.types.type_shape import infer_output_shape, tensor_dimensions
from neurosym.types.type_with_environment import TypeWithEnvironment


class NearTransformer(nn.Module):
    """
    A transformer that works on any types. Will deduce the structure of
    the computation from the context. Also takes into account the environment.
    """

    def __init__(
        self,
        typ,
        max_tensor_size,
        hidden_size,
        num_head,
        num_encoder_layers,
        num_decoder_layers,
    ):
        super().__init__()
        self.typ = typ
        self.max_tensor_size = max_tensor_size
        self.hidden_size = hidden_size
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=num_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True,
        )
        self.pe = BasicMultiDimensionalPositionalEncoding(hidden_size)
        self.proj_in = nn.Linear(max_tensor_size, hidden_size)
        self.proj_out = nn.Linear(
            hidden_size, int(np.prod(tensor_dimensions(self.output_typ)))
        )

    def _flatten_tensor_axes(self, type_shape, x):
        x = x.view(*x.shape[: type_shape.num_batch_and_sequence_axes], -1)
        return x

    def _project_input(self, type_shape, x):
        x = self._flatten_tensor_axes(type_shape, x)
        # add more axes if necessary
        assert (
            x.shape[-1] <= self.max_tensor_size
        ), f"Input tensor too large: {x.shape[-1]} > {self.max_tensor_size}"
        x = torch.cat(
            [
                x,
                torch.zeros(
                    x.shape[:-1] + (self.max_tensor_size - x.shape[-1],),
                    device=x.device,
                    dtype=x.dtype,
                ),
            ],
            dim=-1,
        )
        x = self.proj_in(x)
        return x

    def _output_of_typ(
        self,
        output_typ: Type,
        *inputs: Tuple[TypeAnnotatedObject, ...],
    ):
        assert isinstance(
            output_typ, Type
        ), f"Expected Type, but received {type(output_typ)}"
        type_shape, output_shape = infer_output_shape(
            [x.object_type for x in inputs],
            [x.object_value.shape for x in inputs],
            output_typ,
        )
        inputs = [self._project_input(type_shape, x.object_value) for x in inputs]
        inp = self.pe(type_shape, inputs)
        device = next(self.parameters()).device
        targ = self.pe(
            type_shape,
            [
                torch.zeros(
                    (
                        *output_shape[: type_shape.num_batch_and_sequence_axes],
                        self.hidden_size,
                    ),
                    device=device,
                )
            ],
        )
        out = self.transformer(inp, targ)
        out = self.proj_out(out)
        out = out.view(*output_shape)
        return out

    def forward(self, *args, environment, **kwargs):

        assert (
            not kwargs
        ), f"No keyword arguments are allowed, but received {' '.join(repr(x) for x in kwargs)}"

        if not isinstance(self.typ, ArrowType):
            assert len(args) == 0
            return self._output_of_typ(self.typ, *environment)
        assert len(args) == len(self.typ.input_type)
        return self._output_of_typ(
            self.typ.output_type,
            *[TypeAnnotatedObject(t, x) for x, t in zip(args, self.typ.input_type)],
            *environment,
        )

    @property
    def output_typ(self):
        return self.typ.output_type if isinstance(self.typ, ArrowType) else self.typ


class BasicMultiDimensionalPositionalEncoding(nn.Module):
    """
    Positional encoding that works for multiple dimensions. The idea is that
    the overall positional encoding at a given location is the sum of the
    positional encodings for each dimension. The positional encoding for each
    dimension is computed by taking the base positional encoding and multiplying
    by an orthonormal matrix raised to the power of which dimension it is.

    This is a basic idea and can almost certainly be improved.

    For `forward` this takes a `TypeShape` and a list of tensors, and returns
    a single tensor that is the positional encoding of all the input tensors
    (with the positional encoding added to each input tensor). This is provided
    as a single tensor for ease of use in the transformer.
    """

    def __init__(self, d_model, max_len=10_000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pe = self._get_positional_encoding(max_len, d_model)
        self.orthonormal = nn.Parameter(
            _sample_orthonormal_matrix(d_model), requires_grad=False
        )

    def _get_positional_encoding(self, max_len, d_model):

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe, requires_grad=False)

    def _flatten_batch_axes(self, type_shape, x):
        assert x.shape[: len(type_shape.batch_size)] == type_shape.batch_size
        x = x.view(-1, *x.shape[len(type_shape.batch_size) :])
        return x

    def _positionally_encode_single(self, x):
        """
        Assumes the input has shape (batch_size, *sequence_lengths, d_model).

        Output will have shape (batch_size, *sequence_lengths, d_model)
        and have the semantic property output[batch, loc] = x[batch, loc] + sum_{i < len(loc)} ortho^(i + 1) * pe[loc[i]]
        """
        assert x.shape[-1] == self.d_model
        pe_accum = 0
        ortho = self.orthonormal
        n_seq_axes = len(x.shape) - 2
        for i in range(n_seq_axes):
            pe_this = self.pe[: x.shape[i + 1]] @ ortho
            pe_accum = pe_accum + pe_this
            if i != n_seq_axes - 1:
                pe_accum = pe_accum[..., None, :]
                ortho = ortho @ self.orthonormal
        return x + pe_accum

    def _positionally_encode_all_inputs(self, input_tensors):
        """
        Encode all the input tensors positionally. Encodes each individually, and then
        flattens each, adding a non-orthonormal-multiplied position encooding to each
        depending on its position in the `input_tensors` list.

        Assumes the batch axes have already been flattened.
        """
        input_tensors = [self._positionally_encode_single(x) for x in input_tensors]
        input_tensors = [x.view(x.shape[0], -1, x.shape[-1]) for x in input_tensors]
        pe_each = list(self.pe[: len(input_tensors)])
        input_tensors = [x + pe_each[i] for i, x in enumerate(input_tensors)]
        return torch.cat(input_tensors, dim=1)

    def forward(self, type_shape, input_tensors):
        input_tensors = [self._flatten_batch_axes(type_shape, x) for x in input_tensors]
        inp = self._positionally_encode_all_inputs(input_tensors)
        return inp


def _sample_orthonormal_matrix(d_model):
    """
    Sample an orthonormal matrix of size d_model.
    """
    return torch.nn.init.orthogonal_(torch.empty(d_model, d_model))


class TransformerNeuralHoleFiller(NeuralHoleFiller):
    """
    Hole filler that attempts to use a transformer to fill holes.

    :param hidden_size: The size of the hidden layer in the transformer.
    :param max_tensor_size: The maximum size of the input tensor.
    :param num_head: The number of heads in the transformer.
    :param num_encoder_layers: The number of encoder layers in the transformer.
    :param num_decoder_layers: The number of decoder layers in the transformer.
    """

    def __init__(
        self,
        hidden_size,
        max_tensor_size,
        num_head=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
    ):
        self.hidden_size = hidden_size
        self.max_tensor_size = max_tensor_size
        self.num_head = num_head
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

    def initialize_module(
        self, type_with_environment: TypeWithEnvironment
    ) -> nn.Module | None:
        return NearTransformer(
            max_tensor_size=self.max_tensor_size,
            hidden_size=self.hidden_size,
            num_head=self.num_head,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            typ=type_with_environment.typ,
        )
