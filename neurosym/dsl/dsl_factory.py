from neurosym.dsl.dsl import DSL
from neurosym.dsl.production import ConcreteProduction, ParameterizedProduction
from neurosym.dsl.variable_system import LambdasVariableSystem, NoVariables
from neurosym.types.type_string_repr import TypeDefiner, parse_type
import numpy as np
from neurosym.types.type_signature import expansions, type_universe
from neurosym.types.type import ArrowType, ListType
from neurosym.types.type_string_repr import render_type


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

    def __init__(self, max_expansion_steps=np.inf, max_overall_depth=6, **env):
        self.t = TypeDefiner(**env)
        self._concrete_productions = []
        self._parameterized_productions = []
        self._signatures = []
        self.variable_system = NoVariables()
        self.max_expansion_steps = max_expansion_steps
        self.max_overall_depth = max_overall_depth

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
        sig = self.t.sig(type_str)
        self._concrete_productions.append(
            (
                symbol,
                sig,
                fn,
            )
        )
        self._signatures.append(sig)

    def parameterized(self, symbol, type_str, fn, parameters):
        """
        Add a parameterized production to the DSL.
        """
        sig = self.t.sig(type_str)
        self._parameterized_productions.append(
            (
                symbol,
                sig,
                fn,
                parameters,
            )
        )
        self._signatures.append(sig)

    def _expansions_for_single_production(
        self, expand_to, constructor, symbol, sig, *args
    ):
        sigs = list(
            expansions(
                sig,
                expand_to,
                max_expansion_steps=self.max_expansion_steps,
                max_overall_depth=self.max_overall_depth,
            )
        )
        names = (
            [f"{symbol}_{i}" for i in range(len(sigs))] if len(sigs) > 1 else [symbol]
        )

        assert len(sigs) > 0, f"No expansions within depth/step bounds for {symbol}"

        for name, expansion in zip(names, sigs):
            yield constructor(name, expansion, *args)

    def _expansions_for_all_productions(self, expand_to, constructor, args):
        for arg in args:
            yield from self._expansions_for_single_production(
                expand_to, constructor, *arg
            )

    def finalize(self):
        """
        Finalize the DSL.
        """

        expand_to = type_universe([x.astype() for x in self._signatures])

        productions = []
        productions += self._expansions_for_all_productions(
            expand_to, ConcreteProduction, self._concrete_productions
        )
        productions += self._expansions_for_all_productions(
            expand_to, ParameterizedProduction, self._parameterized_productions
        )

        for p in productions:
            print(p.render())

        print("productions: ", len(productions))

        return DSL(productions, self.variable_system)
