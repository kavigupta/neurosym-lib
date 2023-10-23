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
    signature_expansions,
    type_universe,
)
from neurosym.types.type import ArrowType, AtomicType, ListType, TypeVariable
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
        self._known_types = []
        self._no_zeroadic = False
        self.lambda_parameters = None
        self.max_expansion_steps = max_expansion_steps
        self.max_overall_depth = max_overall_depth
        self.prune = False
        self.target_types = None
        self.prune_variables = False

    def typedef(self, key, type_str):
        """
        Define a type.
        """
        self.t.typedef(key, type_str)

    def known_types(self, *types):
        """
        Add known types to the DSL.
        """
        self._known_types.extend(self.t(typ) for typ in types)

    def no_zeroadic(self):
        """
        Disable zeroadic types (types with no arguments).
        """
        self._no_zeroadic = True

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

    def prune_to(self, *target_types, prune_variables=True):
        """
        Prune the DSL to only include productions that can be constructed from the given
        target types.
        """
        self.prune = True
        self.target_types = [self.t(x) for x in target_types]
        self.prune_variables = prune_variables

    def _expansions_for_single_production(
        self, terminals, type_constructors, constructor, symbol, sig, *args
    ):
        sigs = list(
            signature_expansions(
                sig,
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

        known_types = [x.astype() for x in self._signatures] + self._known_types

        universe = type_universe(known_types, no_zeroadic=self._no_zeroadic)

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

        stable_symbols = set()

        if self.lambda_parameters is not None:
            types, constructors_lambda = type_universe(
                known_types,
                require_arity_up_to=self.lambda_parameters["max_arity"],
                no_zeroadic=self._no_zeroadic,
            )
            top_levels = [
                constructor(
                    *[TypeVariable.fresh() for _ in range(arity - 1)],
                    AtomicType("output_type"),
                )
                for arity, constructor in constructors_lambda
            ]
            top_levels = [x for x in top_levels if isinstance(x, ArrowType)]
            expanded = []
            for top_level in top_levels:
                expanded += expansions(
                    top_level,
                    types,
                    constructors_lambda,
                    max_expansion_steps=self.max_expansion_steps,
                    max_overall_depth=self.lambda_parameters["max_type_depth"],
                )
            expanded = sorted(expanded, key=str)
            sym_to_productions["<lambda>"] = [
                LambdaProduction(i, LambdaTypeSignature(x.input_type))
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
            sym_to_productions["<variable>"] = [
                VariableProduction(
                    type_id, VariableTypeSignature(variable_type, index_in_env)
                )
                for type_id, variable_type in enumerate(variable_types)
                for index_in_env in range(self.lambda_parameters["max_env_depth"])
            ]
            # don't prune and reindex variables
            stable_symbols.add("<variable>")

        if self.prune:
            assert self.target_types is not None
            sym_to_productions = prune(
                sym_to_productions,
                self.target_types,
                care_about_variables=False,
                type_depth_limit=self.max_overall_depth,
                stable_symbols=stable_symbols,
            )
            if self.prune_variables:
                sym_to_productions = prune(
                    sym_to_productions,
                    self.target_types,
                    care_about_variables=True,
                    type_depth_limit=self.max_overall_depth,
                    stable_symbols=stable_symbols,
                )
        if "<variable>" in sym_to_productions:
            sym_to_productions["<variable>"] = clean_variables(
                sym_to_productions["<variable>"]
            )
        dsl = make_dsl(sym_to_productions)
        return dsl


def clean_variables(variable_productions):
    type_to_idx = {prod.type_signature().variable_type for prod in variable_productions}
    type_to_idx = {t: i for i, t in enumerate(sorted(type_to_idx, key=str))}
    variable_productions = [
        prod.with_index(type_to_idx[prod.type_signature().variable_type])
        for prod in variable_productions
    ]
    return variable_productions


def make_dsl(sym_to_productions):
    return DSL([prod for prods in sym_to_productions.values() for prod in prods])


def prune(
    sym_to_productions,
    target_types,
    *,
    care_about_variables,
    type_depth_limit,
    stable_symbols,
):
    dsl = make_dsl(sym_to_productions)
    symbols = dsl.constructible_symbols(
        *target_types,
        care_about_variables=care_about_variables,
        type_depth_limit=type_depth_limit,
    )
    new_sym_to_productions = {}
    for original_symbol, prods in sym_to_productions.items():
        new_sym_to_productions[original_symbol] = [
            x for x in prods if x.symbol() in symbols
        ]
        if len(new_sym_to_productions[original_symbol]) == 0:
            raise TypeError(
                f"All productions for {original_symbol} were pruned. "
                f"Check that the target types are correct."
            )
        if original_symbol in stable_symbols:
            continue
        new_sym_to_productions[original_symbol] = Production.reindex(
            new_sym_to_productions[original_symbol]
        )
    return new_sym_to_productions
