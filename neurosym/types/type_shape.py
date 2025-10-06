import itertools
from dataclasses import dataclass
from typing import Tuple, Union

from neurosym.types.type_string_repr import render_type

from .type import AtomicType, ListType, TensorType, Type


@dataclass
class TypeShape:
    """
    Represents the shape of a tensor with a given type. The idea here is to track the parts
    of the shape that are not explicitly represented in the type. For example, if you have a
    type ``[[{f, 10}]]`` and a shape ``(23, 5, 2, 10)``, you have an implicit batch dimension
    of 23 and sequence dimensions of 5 and 2. This class represents that information in a
    way that can be used to infer the shape of another tensor with the same type.
    """

    batch_size: Tuple[int]
    sequence_lengths: Tuple[int]

    @classmethod
    def check_consistent(cls, *shapes: "TypeShape") -> "TypeShape":
        """
        Combine multiple TypeShapes into a single TypeShape, checking that they are consistent.

        If none are provided return an empty TypeShape.

        :param shapes: The TypeShapes to combine.
        :return: The combined TypeShape.
        """

        assert all(isinstance(shape, TypeShape) for shape in shapes)
        if not shapes:
            return TypeShape(batch_size=(), sequence_lengths=())

        batch_sizes = [shape.batch_size for shape in shapes]
        sequence_lengths = [shape.sequence_lengths for shape in shapes]

        if not all(batch_sizes[0] == batch_size for batch_size in batch_sizes):
            raise TypeShapeException(f"Inconsistent batch sizes: {batch_sizes}")
        if not all(
            sequence_lengths[0] == sequence_length
            for sequence_length in sequence_lengths
        ):
            raise TypeShapeException(
                f"Inconsistent sequence lengths: {sequence_lengths}"
            )

        return TypeShape(
            batch_size=batch_sizes[0], sequence_lengths=sequence_lengths[0]
        )

    def apply(self, typ: Type) -> Tuple[int, ...]:
        """
        Apply the TypeShape to a type, returning the shape of the tensor that would have that type.

        :param typ: The type to apply the TypeShape to.
        :return: The shape of the tensor with the given type.
        """
        dims = _dimension_summary(typ)
        shape = list(self.batch_size)

        count_sequences = dims.count("sequence")
        if count_sequences > len(self.sequence_lengths):
            raise TypeShapeException(
                f"Too few sequence dimensions, {render_type(typ)} expected"
                f" {count_sequences}, got {len(self.sequence_lengths)}"
            )
        if 1 <= count_sequences < len(self.sequence_lengths):
            raise TypeShapeException(
                f"Too many sequence dimensions, {render_type(typ)} expected"
                f" {count_sequences}, got {len(self.sequence_lengths)}"
            )

        sequence_lengths_to_use = list(self.sequence_lengths)
        for dim in dims:
            if dim == "sequence":
                shape.append(sequence_lengths_to_use.pop(0))
            else:
                shape.append(dim)
        assert count_sequences == 0 or not sequence_lengths_to_use
        return tuple(shape)

    @property
    def num_batch_and_sequence_axes(self) -> int:
        """
        Get the number of batch and sequence axes in the TypeShape.

        :return: The number of batch and sequence axes.
        """
        return len(self.batch_size) + len(self.sequence_lengths)

    def squash_batch_axis(self, tensor):
        """
        Reshape a given Tensor with this TypeShape to one with a single batch axis.
        """
        assert self.batch_size == tensor.shape[: len(self.batch_size)]
        return tensor.view((-1, *tensor.shape[len(self.batch_size) :]))

    def unsquash_batch_axis(self, tensor):
        """
        Reshape a given Tensor with this TypeShape to one with the original batch axis.
        """
        return tensor.view((*self.batch_size, *tensor.shape[1:]))

    def reshape_to_nlc(self, tensor):
        """
        Reshape a given Tensor with this TypeShape to one of the form `(N, L, *C)`.
        Requires 0-1 sequence axes.
        """
        assert (
            len(self.sequence_lengths) <= 1
        ), "Can only reshape tensors to NLC if they have 0-1 sequence axes"
        tensor_shape = tensor.shape[self.num_batch_and_sequence_axes :]
        tensor = self.squash_batch_axis(tensor)
        if not self.sequence_lengths:
            tensor = tensor.view((tensor.shape[0], 1, *tensor_shape))
        return tensor


def compute_type_shape(typ: Type, shape: Tuple[int, ...]) -> TypeShape:
    """
    Assign dimensions to the type based on the given shape. E.g., for a type
    ``[[{f, 10}]]`` and shape ``(5, 2, 10)``, the batch size would be ``()`` (no batch dimension),
    and the sequence length would be ``(5, 2)``.

    :param typ: The type to assign dimensions to.
    :param shape: The shape to assign to the type.

    :return: A TypeShape object representing the dimensions of the type.
    """
    dims = _dimension_summary(typ)
    if len(dims) > len(shape):
        raise TypeShapeException(
            f"Too few dimensions for type {render_type(typ)}: expected {len(dims)}, got {len(shape)}: {shape}"
        )
    batch_dims, remaining_shape = (
        shape[: len(shape) - len(dims)],
        shape[len(shape) - len(dims) :],
    )
    assert len(remaining_shape) == len(dims)
    sequence_lengths = []
    for idx, dim_computed, dim_given in zip(itertools.count(), dims, remaining_shape):
        if dim_computed == "sequence":
            sequence_lengths.append(dim_given)
        else:
            if dim_computed != dim_given:
                raise TypeShapeException(
                    f"In type {render_type(typ)}, expected the {idx}th dimension"
                    f" to be {dim_computed}, but received {dim_given} instead: {shape}"
                )
    return TypeShape(
        batch_size=tuple(batch_dims), sequence_lengths=tuple(sequence_lengths)
    )


def _dimension_summary(typ: Type) -> Tuple[Union[int, str]]:
    """
    Compute the dimensions of a type. Each dimension is either a length (for a tensor) or
    "sequence" (for a list).

    :param typ: The type to compute the dimensions of.

    :return: The dimensions of the type.
    """
    assert isinstance(typ, Type), f"Expected Type, but received {type(typ)}"
    dimensions = []
    while True:
        if isinstance(typ, TensorType):
            dimensions.extend(typ.shape)
            if not isinstance(typ.dtype, AtomicType):
                raise TypeShapeException(
                    f"Expected TensorType to have an atomic element typ, but received {render_type(typ.dtype)}"
                )
            break
        if isinstance(typ, AtomicType):
            break
        if isinstance(typ, ListType):
            dimensions.append("sequence")
            typ = typ.element_type
        else:
            raise TypeShapeException(f"Cannot compute shape of {render_type(typ)}")
    return dimensions


def infer_output_shape(
    input_types: Tuple[Type, ...],
    input_shapes: Tuple[Tuple[int, ...], ...],
    output_type: Type,
) -> Tuple[TypeShape, Tuple[int, ...]]:
    """
    Infer the output shape of the transformer given the input shapes.

    The idea here is that you infer the batch axis and sequence axes from the input shapes.
    If there is a disagreement between the inferred batch or sequence axes, an error is raised.

    :param input_types: Types of the input tensors.
    :param input_shapes: Shapes of the input tensors.
    :param output_type: Type of the output tensor.
    """
    assert len(input_types) == len(input_shapes)

    dimensions = TypeShape.check_consistent(
        *[compute_type_shape(t, s) for t, s in zip(input_types, input_shapes)]
    )
    return dimensions, dimensions.apply(output_type)


def tensor_dimensions(typ: Type) -> Tuple[int, ...]:
    """
    Compute the tensor dimensions of a given type

    :param typ: The type to compute the tensor dimensions of, e.g., ``[[{f, 10, 20}]]``.
    :return: The tensor dimensions of the type, e.g., ``(10, 20)``.
    """
    dims = _dimension_summary(typ)
    return tuple(dim for dim in dims if dim != "sequence")


class TypeShapeException(Exception):
    """
    An error raised when there is an issue with computing a TypeShape.
    """
