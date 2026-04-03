from dataclasses import dataclass
from typing import List, Tuple

from neurosym.dsl.dsl import DSL
from neurosym.programs.s_expression import InitializedSExpression
from neurosym.types.type import Type
from neurosym.types.type_annotated_object import TypeAnnotatedObject


@dataclass
class LambdaFunction:
    dsl: DSL
    body: InitializedSExpression
    input_types: Tuple[Type, ...]
    parent_environment: List[object]

    @classmethod
    def of(cls, *, dsl, body, input_types, parent_environment):
        """Create a LambdaFunction."""
        return cls(dsl, body, input_types, parent_environment)

    def __call__(self, *args):
        assert len(args) == len(self.input_types)
        # Reverse the arguments because we want to number them from the right.
        type_annotated_args = [
            TypeAnnotatedObject(arg_type, arg)
            for arg_type, arg in zip(self.input_types, args)
        ]
        return self.dsl.compute(
            self.body, [*type_annotated_args[::-1], *self.parent_environment]
        )
