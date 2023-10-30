from abc import ABC, abstractmethod
import numpy as np
import torch
from neurosym.types.type import (
    ArrowType,
    ListType,
    TensorType,
    AtomicType,
)
from neurosym.types.type_string_repr import render_type


class TypeCheckError(Exception):
    def __init__(self, msg, context):
        super().__init__("\n".join([msg] + [f"\twhen {x}" for x in context[::-1]]))


class TypeChecker(ABC):
    @abstractmethod
    def construct_tensor(self, data):
        pass

    @abstractmethod
    def tensor_type(self):
        pass

    @abstractmethod
    def tensor_shape_and_element(self, tensor):
        pass

    def construct_list(self, data):
        return [data] * 3

    @abstractmethod
    def exemplar_for_atomic(self, name, *, context):
        pass

    @abstractmethod
    def check_atomic_type(self, name, val):
        pass

    def output_exemplar(self, typ, args, *, context):
        return self.exemplar(
            typ,
            context=(
                *context,
                f"constructing an exemplar for the output of type {render_type(typ)}",
            ),
        )

    def arrow_exemplar(self, typ, *, context):
        context = (*context, f"constructing arrow exemplar of type {render_type(typ)}")
        assert isinstance(typ, ArrowType)

        def f(*args):
            if len(args) != len(typ.input_type):
                raise TypeCheckError(
                    f"In {render_type(typ)}: Expected {len(typ.input_type)} arguments, got {len(args)}",
                    context,
                )
            for i, (arg, arg_typ) in enumerate(zip(args, typ.input_type)):
                self.typecheck(
                    arg_typ, arg, context=(*context, f"checking argument {i}")
                )
            return self.output_exemplar(
                typ.output_type,
                args,
                context=(
                    *context,
                    f"constructing the output exemplar of type {render_type(typ.output_type)}",
                ),
            )

        return f

    def exemplar(self, typ, *, context):
        if isinstance(typ, ArrowType):
            return self.arrow_exemplar(typ, context=context)
        if isinstance(typ, ListType):
            return self.construct_list(
                self.exemplar(
                    typ.element_type,
                    context=(
                        *context,
                        f"constructing an exemplar for the list content of type {render_type(typ.element_type)}",
                    ),
                )
            )
        if isinstance(typ, TensorType):
            result = self.exemplar(
                typ.dtype,
                context=(
                    *context,
                    f"constructing an exemplar for the tensor content of type {render_type(typ.dtype)}",
                ),
            )
            for dim in typ.shape[::-1]:
                result = [result] * dim
            return self.construct_tensor(result)
        if isinstance(typ, AtomicType):
            return self.exemplar_for_atomic(typ.name, context=context)
        raise NotImplementedError(f"Exemplar not implemented for {typ}")

    def typecheck(self, typ, val, *, context):
        if isinstance(typ, ArrowType):
            exemplars = [
                self.exemplar(
                    inp_typ,
                    context=(
                        *context,
                        f"constructing an exemplar for the {i}th arrow type input of type {render_type(typ)}",
                    ),
                )
                for i, inp_typ in enumerate(typ.input_type)
            ]
            print(exemplars)
            try:
                out = val(*exemplars)
            except Exception as e:
                raise TypeCheckError(
                    f"Exception while calling {render_type(typ)}: {e}", context
                )
            self.typecheck(
                typ.output_type,
                out,
                context=(
                    *context,
                    f"checking output of {val} to have type {render_type(typ.output_type)}",
                ),
            )
            return
        if isinstance(typ, ListType):
            for v in self.iterate_list(val):
                self.typecheck(
                    typ.element_type,
                    v,
                    context=(
                        *context,
                        f"checking list element to have type {render_type(typ.element_type)}",
                    ),
                )
            return
        if isinstance(typ, TensorType):
            print(render_type(typ), val.shape)
            if not isinstance(val, self.tensor_type()):
                raise TypeCheckError(f"Expected tensor, got {val}", context)
            val_shape, val = self.tensor_shape_and_element(val)
            if val_shape != typ.shape:
                raise TypeCheckError(
                    f"Expected tensor of shape {typ.shape}, got {val_shape}", context
                )
            self.typecheck(
                typ.dtype, val, context=(*context, "checking tensor content")
            )
            return
        if isinstance(typ, AtomicType):
            self.check_atomic_type(typ.name, val, context=context)
            return
        raise NotImplementedError(f"Typecheck not implemented for {typ}")


class TorchTypeChecker(TypeChecker):
    torch_types = {
        "i": torch.int,
        "f": torch.float,
        "b": torch.bool,
    }

    def construct_tensor(self, data):
        return torch.tensor(np.array(data))[None]

    def tensor_type(self):
        return torch.Tensor

    def tensor_shape_and_element(self, tensor):
        return tensor.shape[1:], tensor.flatten()[0]

    def construct_list(self, data):
        return torch.stack([data] * 3).transpose(0, 1)

    def exemplar_for_atomic(self, name, *, context):
        if name in self.torch_types:
            return torch.tensor(0, dtype=self.torch_types[name])
        raise TypeCheckError(f"Unknown torch atomic type {name}", context)

    def check_atomic_type(self, name, val, *, context):
        if name in self.torch_types:
            if val.dtype != self.torch_types[name]:
                raise TypeCheckError(f"Expected {name}, got {val.dtype}", context)
            return
        raise TypeCheckError(f"Unknown torch atomic type {name}", context)

    def output_exemplar(self, typ, args, *, context):
        if not isinstance(typ, TensorType):
            return super().output_exemplar(typ, args, context=context)

        tensor_args = [x for x in args if isinstance(x, torch.Tensor)]
        if len(tensor_args) == 0:
            return super().output_exemplar(typ, args, context=context)
        [num_batches] = {x.shape[0] for x in tensor_args}
        exemplar = super().output_exemplar(typ, args, context=context)
        print(exemplar.shape)
        exemplar = exemplar.repeat(num_batches, *([1] * (len(typ.shape) + 1)))
        print(exemplar.shape)
        return exemplar
