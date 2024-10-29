import copy
from typing import Callable, Dict, List, Tuple

import numpy as np

from ..types.type import ArrowType, AtomicType, Type, TypeVariable
from ..types.type_signature import (
    FunctionTypeSignature,
    LambdaTypeSignature,
    VariableTypeSignature,
    _signature_expansions,
    _type_universe,
    type_expansions,
)
from ..types.type_string_repr import TypeDefiner
from .dsl import DSL
from .production import (
    ConcreteProduction,
    LambdaProduction,
    ParameterizedProduction,
    Production,
    VariableProduction,
)


class DSLFactory:
    """
    A factory for creating DSLs.

    Example usage:

    .. highlight:: python
    .. code-block:: python

        dslf = DSLFactory()
        dslf.typedef("fn", "(i) -> i")
        dslf.concrete("inc", "$fn", lambda x: x + 1)
        dslf.concrete("const_0", "$fn", lambda x: 0)
        dslf.concrete("compose", "($fn, $fn) -> $fn", lambda f, g: lambda x: f(g(x))
        dslf.finalize()
    """

    def __init__(
        self, max_expansion_steps=np.inf, max_env_depth=4, max_overall_depth=6, **env
    ):
        self.t = TypeDefiner(**env)
        self._concrete_productions = []
        self._parameterized_productions = []
        self._signatures = []
        self._known_types = []
        self._no_zeroadic = False
        self.lambda_parameters = None
        self.max_expansion_steps = max_expansion_steps
        self.max_overall_depth = max_overall_depth
        self.max_env_depth = max_env_depth
        self.prune = False
        self.target_types = None
        self.prune_variables = False
        self.tolerate_pruning_entire_productions = False

    def typedef(self, key: str, type_str: str):
        """
        Define a type with the given type string.
        The key will be used to refer to the type in future calls
        with a $ prefix. E.g.,

        .. highlight:: python
        .. code-block:: python

            dslf.typedef("fn", "(i) -> i")
            dslf.concrete("inc", "$fn", lambda x: x + 1)
        """
        self.t.typedef(key, type_str)

    def filtered_type_variable(self, key, type_filter: Callable[[Type], bool]):
        """
        Define a filtered type variable. This is a type variable that can only be
        instantiated with types that satisfy the given filter. The key will be used to
        refer to the type in future calls with a % prefix. E.g.,

        .. highlight:: python
        .. code-block:: python

            dslf.filtered_type_variable(
                "num", lambda x: isinstance(x, ns.AtomicType) and x.name in ["i", "f"]
            )
            dslf.concrete("+", "%num -> %num -> %num", lambda x: x)
        """
        self.t.filtered_type_variable(key, type_filter)

    def known_types(self, *types: Tuple[str, ...]):
        """
        Make this DSLFactory aware of the given types. These types will be used to
        generate expansions for any productions need to be template-expanded.
        """
        self._known_types.extend(self.t(typ) for typ in types)

    def no_zeroadic(self):
        """
        Disable zeroadic types (types with no arguments).
        """
        self._no_zeroadic = True

    def lambdas(self, max_arity=2, max_type_depth=4, max_env_depth=4):
        """
        Add lambda productions to the DSL. This will add (lam_0, lam_1, ..., lam_n)
        productions for each argument type/arity combination, as well as
        ($i_j) productions for each variable de bruijn index i and type j.

        :param max_arity: The maximum arity of lambda functions to generate.
        :param max_type_depth: The maximum depth of types to generate.
        :param max_env_depth: The maximum depth of the environment to generate.
        """
        self.lambda_parameters = dict(
            max_arity=max_arity,
            max_type_depth=max_type_depth,
            max_env_depth=max_env_depth,
        )

    def concrete(self, symbol: str, type_str: str, semantics: object):
        """
        Add a concrete production to the DSL.

        :param symbol: The symbol for the production.
        :param type_str: The type string for the production.
        :param semantics: The semantics to use for the production. This should have
            a type corresponding to ``type_str``. Note: *this is not checked*.
        """
        sig = self.t.sig(type_str)
        self._concrete_productions.append(
            (
                symbol,
                sig,
                semantics,
            )
        )
        self._signatures.append(sig)

    def parameterized(
        self,
        symbol: str,
        type_str: str,
        semantics: object,
        parameters: Dict[str, Callable[[], object]],
    ):
        """
        Add a parameterized production to the DSL.

        :param symbol: The symbol for the production.
        :param type_str: The type string for the production.
        :param semantics: The semantics to use for the production. This should have
            a type corresponding to ``type_str``. Note: *this is not checked*.
        :param parameters: A dictionary mapping parameter names to functions that
            generate initial parameter values.
        """
        sig = self.t.sig(type_str)
        self._parameterized_productions.append(
            (
                symbol,
                sig,
                semantics,
                parameters,
            )
        )
        self._signatures.append(sig)

    def prune_to(
        self,
        *target_types: Tuple[str, ...],
        prune_variables=True,
        tolerate_pruning_entire_productions=False,
    ):
        """
        Direct the current DSLFactory to prune any productions p such that there does not exist some
        program s and type t in target_types such that s :: t and s contains p as a production.
        """
        self.prune = True
        self.target_types = [self.t(x) for x in target_types]
        self.prune_variables = prune_variables
        self.tolerate_pruning_entire_productions = tolerate_pruning_entire_productions

    def _expansions_for_single_production(
        self, terminals, type_constructors, constructor, symbol, sig, *args
    ):
        sigs = list(
            _signature_expansions(
                sig,
                terminals,
                type_constructors,
                max_expansion_steps=self.max_expansion_steps,
                max_overall_depth=self.max_overall_depth,
            )
        )
        assert len(sigs) > 0, f"No expansions within depth/step bounds for {symbol}"

        prods = [
            constructor(symbol, FunctionTypeSignature.from_type(expansion), *args)
            for expansion in sigs
        ]

        return {symbol: Production.reindex(prods)}

    def _expansions_for_all_productions(
        self, expand_to, terminals, type_constructors, args
    ):
        result = {}
        for arg in args:
            for_prod = self._expansions_for_single_production(
                expand_to, terminals, type_constructors, *arg
            )
            duplicate_keys = sorted(set(for_prod.keys()) & set(result.keys()))
            if duplicate_keys:
                for key in duplicate_keys:
                    if for_prod[key] != result[key]:
                        raise ValueError(
                            f"Duplicate declarations for production: {key}"
                        )
            else:
                result.update(for_prod)
        return result

    def finalize(self) -> DSL:
        """
        Produce the DSL from this factory. This will generate all productions and
        potentially raise errors if there were issues with the way the DSL was
        constructed.
        """

        known_types = [x.astype() for x in self._signatures] + self._known_types

        universe = _type_universe(known_types, no_zeroadic=self._no_zeroadic)

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
            types, constructors_lambda = _type_universe(
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
                expanded += type_expansions(
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
            sym_to_productions = _prune(
                sym_to_productions,
                self.target_types,
                care_about_variables=False,
                type_depth_limit=self.max_overall_depth,
                env_depth_limit=self.max_env_depth,
                stable_symbols=stable_symbols,
                tolerate_pruning_entire_productions=self.tolerate_pruning_entire_productions,
            )
            if self.prune_variables:
                sym_to_productions = _prune(
                    sym_to_productions,
                    self.target_types,
                    care_about_variables=True,
                    type_depth_limit=self.max_overall_depth,
                    env_depth_limit=self.max_env_depth,
                    stable_symbols=stable_symbols,
                    tolerate_pruning_entire_productions=self.tolerate_pruning_entire_productions,
                )
        if "<variable>" in sym_to_productions:
            sym_to_productions["<variable>"] = _clean_variables(
                sym_to_productions["<variable>"]
            )
        dsl = _make_dsl(
            sym_to_productions,
            copy.copy(self.target_types),
            self.max_overall_depth,
            self.max_env_depth,
        )
        return dsl


def _clean_variables(variable_productions):
    type_to_idx = {prod.type_signature().variable_type for prod in variable_productions}
    type_to_idx = {t: i for i, t in enumerate(sorted(type_to_idx, key=str))}
    variable_productions = [
        prod.with_index(type_to_idx[prod.type_signature().variable_type])
        for prod in variable_productions
    ]
    return variable_productions


def _make_dsl(sym_to_productions, valid_root_types, max_type_depth, max_env_depth):
    return DSL(
        [prod for prods in sym_to_productions.values() for prod in prods],
        valid_root_types,
        max_type_depth,
        max_env_depth=max_env_depth,
    )


def _prune(
    sym_to_productions,
    target_types,
    *,
    care_about_variables,
    type_depth_limit,
    env_depth_limit,
    stable_symbols,
    tolerate_pruning_entire_productions,
):
    dsl = _make_dsl(sym_to_productions, target_types, type_depth_limit, env_depth_limit)
    symbols = dsl.constructible_symbols(care_about_variables=care_about_variables)
    new_sym_to_productions = {}
    for original_symbol, prods in sym_to_productions.items():
        new_sym_to_productions[original_symbol] = [
            x for x in prods if x.symbol() in symbols
        ]
        if (
            len(new_sym_to_productions[original_symbol]) == 0
            and not tolerate_pruning_entire_productions
        ):
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
