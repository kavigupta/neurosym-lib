from neurosym.dsl.dsl import DSL
from neurosym.dsl.production import ConcreteProduction, ParameterizedProduction
from neurosym.dsl.variable_system import LambdasVariableSystem, NoVariables
from neurosym.types.type_string_repr import TypeDefiner


class DSLFactory:
    """
    A factory for creating DSLs.

    Example usage:
    ```
    factory = DSLFactory()
    factory.typedef("fn", "(i) -> i")
    factory.concrete("inc", "$fn", lambda x: x + 1)
    factory.concrete("const_0", "$fn", lambda x: 0)
    factory.concrete("compose", "($fn, $fn) -> $fn", lambda f, g: lambda x: f(g(x))
    factory.finalize()
    """

    def __init__(self, **env):
        self.t = TypeDefiner(**env)
        self.productions = []
        self.variable_system = NoVariables()

    def typedef(self, key, type_str):
        """
        Define a type.
        """
        self.t.typedef(key, type_str)

    def lambdas(self, lambda_arity_limit: int = 2, num_variable_limit: int = 3):
        self.variable_system = LambdasVariableSystem(
            lambda_arity_limit=lambda_arity_limit, num_variable_limit=num_variable_limit
        )

    def concrete(self, symbol, type_str, fn):
        """
        Add a concrete production to the DSL.
        """
        self.productions.append(
            ConcreteProduction(
                symbol,
                self.t.sig(type_str),
                fn,
            )
        )

    def parameterized(self, symbol, type_str, fn, parameters):
        """
        Add a parameterized production to the DSL.
        """
        self.productions.append(
            ParameterizedProduction(
                symbol,
                self.t.sig(type_str),
                fn,
                parameters,
            )
        )

    def finalize(self):
        """
        Finalize the DSL.
        """
        return DSL(self.productions, self.variable_system)
