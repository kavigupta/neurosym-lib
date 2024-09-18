import math
from typing import Tuple

import torch
import torch.nn as nn

from neurosym.types.type import ArrowType, Type
from neurosym.types.type_annotated_object import TypeAnnotatedObject
from neurosym.types.type_shape import infer_output_shape


class NearTransformer(nn.Module):
    """
    A transformer that works on any types. Will deduce the structure of
    the computation from the context. Also takes into account the environment.
    """
    def __init__(
        self, typ, hidden_size, num_head, num_encoder_layers, num_decoder_layers
    ):
        super().__init__()
        self.typ = typ
        self.hidden_size = hidden_size
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=num_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True,
        )
        self.pe = BasicMultiDimensionalPositionalEncoding(hidden_size)
        self.proj_in = nn.Linear(1, hidden_size)
        self.proj_out = nn.Linear(hidden_size, 1)

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
        inp = self.pe(
            type_shape, [self.proj_in(x.object_value[..., None]) for x in inputs]
        )
        device = next(self.parameters()).device
        targ = self.pe(
            type_shape, [torch.zeros((*output_shape, self.hidden_size), device=device)]
        )
        out = self.transformer(inp, targ)
        out = self.proj_out(out)
        out = out.view(*output_shape)
        return out

    def forward(self, *args, environment, **kwargs):

        assert (
            not kwargs
        ), f"No keyword arguments are allowed, but received {' '.join(repr(x) for x in kwargs.keys())}"

        if isinstance(self.typ, ArrowType):
            assert len(args) == len(self.typ.input_type)
            return self._output_of_typ(
                self.typ.output_type,
                *[TypeAnnotatedObject(t, x) for x, t in zip(args, self.typ.input_type)],
                *environment,
            )
        else:
            assert len(args) == 0
            return self._output_of_typ(self.typ, *environment)


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

        Output will have shape (batch_size, sequence_lengths, d_model)
        and have the semantic property output[batch, loc] = x[batch, loc] + sum_{i < len(loc)} ortho^(i + 1) * pe[loc[i]]
        """
        assert x.shape[-1] == self.d_model
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        pe_accum = 0
        ortho = self.orthonormal
        for i in range(len(x.shape) - 2):
            pe_this = self.pe[: x.shape[i + 1]] @ ortho
            pe_accum = pe_accum + pe_this
            pe_accum = pe_accum[..., None, :]
            ortho = ortho @ self.orthonormal
        pe_accum = pe_accum.squeeze(-2)
        return x + pe_accum

    def _positionally_encode_all_inputs(self, input_tensors):
        """
        Encode all the input tensors positionally. Encodes each individually, and then
        flattens each, adding a non-orthonormal-multiplied position encooding to each
        depending on its position in the `input_tensors` list.

        Assumes the batch axes have already been flattened.
        """
        # print([x.shape for x in input_tensors])
        input_tensors = [self._positionally_encode_single(x) for x in input_tensors]
        # print([x.shape for x in input_tensors])
        input_tensors = [x.view(x.shape[0], -1, x.shape[-1]) for x in input_tensors]
        # print([x.shape for x in input_tensors])
        pe_each = list(self.pe[: len(input_tensors)])
        input_tensors = [x + pe_each[i] for i, x in enumerate(input_tensors)]
        # print([x.shape for x in input_tensors])
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


def transformer_factory(
    hidden_size: int,
    num_head: int = 8,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
):
    """
    Allows instantiating a transformer with a given hidden size, number of heads,

    This transformer should work on any types, and will deduce the structure of
    the computation from the context. Also takes into account the environment.

    :param hidden_size: The hidden size of the transformer (d_model).
    :param num_head: The number of heads in the transformer.
    :param num_encoder_layers: The number of encoder layers in the transformer.
    :param num_decoder_layers: The number of decoder layers in the transformer.
    """

    def construct_model(typ):
        return NearTransformer(
            hidden_size=hidden_size,
            num_head=num_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            typ=typ,
        )

    return construct_model
