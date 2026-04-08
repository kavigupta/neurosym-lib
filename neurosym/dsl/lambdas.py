from dataclasses import dataclass
from typing import List

from neurosym.dsl.dsl import DSL
from neurosym.programs.s_expression import InitializedSExpression
from neurosym.types.type import TypeVariable
from neurosym.types.type_annotated_object import TypeAnnotatedObject
from neurosym.types.type_signature import LambdaTypeSignature


@dataclass
class LambdaFunction:
    dsl: DSL
    body: InitializedSExpression
    typ: LambdaTypeSignature
    parent_environment: List[object]
    _input_types: object = None  # lazily computed

    @classmethod
    def of(cls, dsl, body, typ, parent_environment):
        return cls(dsl, body, typ, parent_environment)

    @property
    def input_types(self):
        """Lazily compute the input types from the body's type."""
        if self._input_types is None:
            twe = self.dsl.compute_type(self.body)
            env = twe.env
            arity = self.typ.function_arity
            # Extract types from the body's environment (indices 0..arity-1)
            types = []
            for i in range(arity):
                if hasattr(env, "_elements") and i in env._elements:
                    types.append(env._elements[i])
                else:
                    types.append(TypeVariable(f"__lam_arg_{i}"))
            self._input_types = tuple(types)
        return self._input_types

    def __call__(self, *args):
        assert len(args) == self.typ.function_arity
        # Reverse the arguments because we want to number them from the right.
        type_annotated_args = [
            TypeAnnotatedObject(t, arg) for t, arg in zip(self.input_types, args)
        ]
        return self.dsl.compute(
            self.body, [*type_annotated_args[::-1], *self.parent_environment]
        )
