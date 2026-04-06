from dataclasses import dataclass
from typing import List, Optional, Union

from torch import NoneType

from neurosym.types.type import TypeVariable
from neurosym.types.type_signature import TypeSignature
from neurosym.types.type_with_environment import TypeWithEnvironment


@dataclass
class ShieldTypeSignature(TypeSignature):
    """
    Represents the type signature of the shield production. A shield operation
    shields its argument from the particular variable in the environment.

    This is parameterized by the type of the variable being removed, so
    we can infer forward typing.

    :param index_in_env: The index of the variable in the environment.
    """

    index_in_env: int

    def arity(self) -> int:
        # just the body
        return 1

    def render(self) -> str:
        # pylint: disable=cyclic-import
        from neurosym.types.type_string_repr import render_type

        body = TypeVariable("body")

        return f"D<{render_type(body)}, ${self.index_in_env}> -> {render_type(body)}"

    def required_env_index(self) -> Optional[int]:
        return self.index_in_env

    def unify_return(
        self, twe: TypeWithEnvironment
    ) -> Union[List[TypeWithEnvironment], NoneType]:
        new_env = twe.env.attempt_remove(self.index_in_env)
        if new_env is None:
            return None
        return [TypeWithEnvironment(twe.typ, new_env)]

    def return_type_template(self):
        return TypeVariable("body")

    def unify_arguments(
        self, twes: List[TypeWithEnvironment]
    ) -> Union[TypeWithEnvironment, NoneType]:
        if len(twes) != 1:
            return None
        twe = twes[0]
        new_env = twe.env.attempt_insert(self.index_in_env)
        if new_env is None:
            return None
        return TypeWithEnvironment(twe.typ, new_env)
