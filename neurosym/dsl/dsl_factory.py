from typing import Dict, List
from neurosym.dsl.dsl import DSL
from neurosym.dsl.production import (
    ConcreteProduction,
    LambdaProduction,
    ParameterizedProduction,
    VariableProduction,
    Production,
)
from neurosym.types.type_string_repr import TypeDefiner, parse_type
import numpy as np
from neurosym.types.type_signature import (
    ConcreteTypeSignature,
    LambdaTypeSignature,
    VariableTypeSignature,
    expansions,
    type_universe,
)
from neurosym.types.type import ArrowType, ListType, TypeVariable
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
        self.lambda_parameters = None
        self.max_expansion_steps = max_expansion_steps
        self.max_overall_depth = max_overall_depth

    def typedef(self, key, type_str):
        """
        Define a type.
        """
        self.t.typedef(key, type_str)

    def lambdas(self, max_arity=2, max_type_depth=4, max_env_depth=4):
        """
        Define a type.
        """
        self.lambda_parameters = dict(
            max_arity=max_arity,
            max_type_depth=max_type_depth,
            max_env_depth=max_env_depth,
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
        self, terminals, type_constructors, constructor, symbol, sig, *args
    ):
        sigs = list(
            expansions(
                sig.astype(),
                terminals,
                type_constructors,
                max_expansion_steps=self.max_expansion_steps,
                max_overall_depth=self.max_overall_depth,
            )
        )
        assert len(sigs) > 0, f"No expansions within depth/step bounds for {symbol}"

        prods = [
            constructor(symbol, ConcreteTypeSignature.from_type(expansion), *args)
            for expansion in sigs
        ]

        return {symbol: Production.reindex(prods)}

    def _expansions_for_all_productions(
        self, expand_to, terminals, type_constructors, args
    ):
        result = {}
        for arg in args:
            result.update(
                self._expansions_for_single_production(
                    expand_to, terminals, type_constructors, *arg
                )
            )
        return result

    def finalize(self):
        """
        Finalize the DSL.
        """

        known_types = [x.astype() for x in self._signatures]

        universe = type_universe(known_types)

        sym_to_productions: Dict[str, List[Production]] = {}
        sym_to_productions.update(
            self._expansions_for_all_productions(
                *universe, ConcreteProduction, self._concrete_productions
            )
        )
        sym_to_productions.update(
            self._expansions_for_all_productions(
                *universe, ParameterizedProduction, self._parameterized_productions
            )
        )

        if self.lambda_parameters is not None:
            universe_lambda = type_universe(
                known_types, require_arity_up_to=self.lambda_parameters["max_arity"]
            )
            expanded = expansions(
                TypeVariable("ROOT"),
                *universe_lambda,
                max_expansion_steps=self.max_expansion_steps,
                max_overall_depth=self.lambda_parameters["max_type_depth"],
            )
            expanded = list(expanded)
            expanded = [x for x in expanded if isinstance(x, ArrowType)]
            sym_to_productions["<lambda>"] = [
                LambdaProduction(i, LambdaTypeSignature(x))
                for i, x in enumerate(expanded)
            ]

            variable_types = sorted(
                {
                    input_type
                    for function_type in expanded
                    for input_type in function_type.input_type
                },
                key=str,
            )
            variable_type_signatures = [
                VariableTypeSignature(variable_type, index_in_env)
                for variable_type in variable_types
                for index_in_env in range(self.lambda_parameters["max_env_depth"])
            ]
            sym_to_productions["<variable>"] = [
                VariableProduction(unique_id, variable_type_sig)
                for unique_id, variable_type_sig in enumerate(variable_type_signatures)
            ]

        return DSL([prod for prods in sym_to_productions.values() for prod in prods])
