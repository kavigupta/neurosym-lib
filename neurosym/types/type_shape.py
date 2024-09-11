from dataclasses import dataclass
from typing import Tuple, Union

from .type import AtomicType, ListType, TensorType, Type


@dataclass
class TypeShape:
    batch_size: Tuple[int]
    sequence_lengths: Tuple[int]

    @classmethod
    def check_consistent(cls, *shapes: "TypeShape") -> "TypeShape":
        """
        Combine multiple TypeShapes into a single TypeShape, checking that they are consistent.

        If none are provided return an empty TypeShape.
        """

        assert all(isinstance(shape, TypeShape) for shape in shapes)
        if not shapes:
            return TypeShape(batch_size=(), sequence_lengths=())

        batch_sizes = [shape.batch_size for shape in shapes]
        sequence_lengths = [shape.sequence_lengths for shape in shapes]

        if not all(batch_sizes[0] == batch_size for batch_size in batch_sizes):
            raise TypeShapeError("Inconsistent batch sizes")
        if not all(
            sequence_lengths[0] == sequence_length
            for sequence_length in sequence_lengths
        ):
            raise TypeShapeError("Inconsistent sequence lengths")

        return TypeShape(
            batch_size=batch_sizes[0], sequence_lengths=sequence_lengths[0]
        )

    def apply(self, typ: Type) -> Tuple[int, ...]:
        """
        Apply the TypeShape to a type, returning the shape of the tensor that would have that type.
        """
        dims = dimension_summary(typ)
        shape = list(self.batch_size)
        sequence_lengths_to_use = list(self.sequence_lengths)[::-1]
        for dim in dims[::-1]:
            if dim == "sequence":
                if not sequence_lengths_to_use:
                    raise TypeShapeError("Too many sequence types")
                shape.append(sequence_lengths_to_use.pop())
            else:
                shape.append(dim)
        if sequence_lengths_to_use:
            raise TypeShapeError("Not enough sequence types")
        return tuple(shape)


def compute_type_shape(typ: Type, shape: Tuple[int, ...]) -> TypeShape:
    """
    Assign dimensions to the type based on the given shape. E.g., for a type
    `[[{f, 10}]]` and shape `(5, 2, 10)`, the batch size would be () (no batch dimension),
    and the sequence length would be (5, 2).

    :param typ: The type to assign dimensions to.
    :param shape: The shape to assign to the type.

    :return: A TypeShape object representing the dimensions of the type.
    """
    dims = dimension_summary(typ)
    if len(dims) > len(shape):
        raise TypeShapeError(
            f"Too few dimensions for type {typ}: expected {len(dims)}, got {len(shape)}"
        )
    batch_dims, remaining_shape = (
        shape[: len(shape) - len(dims)],
        shape[len(shape) - len(dims) :],
    )
    assert len(remaining_shape) == len(dims)
    sequence_lengths = []
    for dim_computed, dim_given in zip(dims, remaining_shape):
        if dim_computed == "sequence":
            sequence_lengths.append(dim_given)
        else:
            if dim_computed != dim_given:
                raise TypeShapeError(
                    f"Expected dimension {dim_computed} to be {dim_given}"
                )
    return TypeShape(
        batch_size=tuple(batch_dims), sequence_lengths=tuple(sequence_lengths)
    )


def dimension_summary(typ: Type) -> Tuple[Union[int, str]]:
    """
    Compute the dimensions of a type. Each dimension is either a length (for a tensor) or
    "sequence" (for a list).

    :param typ: The type to compute the dimensions of.

    :return: The dimensions of the type.
    """
    dimensions = []
    while True:
        if isinstance(typ, TensorType):
            dimensions.extend(typ.shape)
            if not isinstance(typ.dtype, AtomicType):
                raise TypeShapeError(
                    f"Expected TensorType to have an atomic element typ, but received {typ}"
                )
            break
        elif isinstance(typ, ListType):
            dimensions.append("sequence")
            typ = typ.element_type
        else:
            raise TypeShapeError(f"Cannot compute shape of {typ}")
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


class TypeShapeError(Exception):
    pass
