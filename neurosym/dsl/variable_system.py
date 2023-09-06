from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple
from neurosym.programs.hole import Hole
from neurosym.programs.s_expression import SExpression

from neurosym.types.type import ArrowType, Type
from neurosym.types.type_with_environment import TypeWithEnvironment


class VariableSystem(ABC):
    """
    Represents a system for injecting variables and lambdas.
    """

    @abstractmethod
    def lambdas_for_type(self, twe):
        pass


class NoVariables(VariableSystem):
    def lambdas_for_type(self, twe):
        return []


@dataclass
class LambdasVariableSystem(VariableSystem):

    lambda_arity_limit: int
    num_variable_limit: int

    def lambdas_for_type(self, twe):
        typ = twe.typ
        if not isinstance(typ, ArrowType):
            return []
        args = typ.input_type
        if len(args) > self.lambda_arity_limit:
            return []

        new_environment = twe.env.child(*args)
        if len(new_environment) > self.num_variable_limit:
            return []

        return [
            SExpression(
                "lam" + str(len(args)),
                (Hole.of(TypeWithEnvironment(typ.output_type, new_environment)),),
            )
        ]
