from dataclasses import dataclass
from typing import List

from neurosym.dsl.dsl import DSL
from neurosym.programs.s_expression import InitializedSExpression
from neurosym.types.type_annotated_object import TypeAnnotatedObject
from neurosym.types.type_signature import LambdaTypeSignature


@dataclass
class LambdaFunction:
    dsl: DSL
    body: InitializedSExpression
    typ: LambdaTypeSignature
    parent_environment: List[object]

    @classmethod
    def of(cls, dsl, body, typ, parent_environment):
        return cls(dsl, body, typ, parent_environment)

    def __call__(self, *args):
        assert len(args) == self.typ.function_arity()
        # Reverse the arguments because we want to number them from the right.
        type_annotated_args = [
            TypeAnnotatedObject(arg_type, arg)
            for arg_type, arg in zip(self.typ.input_types, args)
        ]
        return self.dsl.compute(
            self.body, [*type_annotated_args[::-1], *self.parent_environment]
        )
