import copy
import warnings
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
        dslf.production("inc", "$fn", lambda x: x + 1)
        dslf.production("const_0", "$fn", lambda x: 0)
        dslf.production("compose", "($fn, $fn) -> $fn", lambda f, g: lambda x: f(g(x))
        dslf.finalize()
    """

    def __init__(
        self, max_expansion_steps=np.inf, max_env_depth=4, max_overall_depth=6, **env
    ):
        self.t = TypeDefiner(**env)
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
        self._extra_productions = []

    def typedef(self, key: str, type_str: str):
        """
        Define a type with the given type string.
        The key will be used to refer to the type in future calls
        with a $ prefix. E.g.,

        .. highlight:: python
        .. code-block:: python

            dslf.typedef("fn", "(i) -> i")
            dslf.production("inc", "$fn", lambda x: x + 1)
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
            dslf.production("+", "%num -> %num -> %num", lambda x: x)
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

    def lambdas(self, max_type_depth=4):
        """
        Add lambda productions to the DSL. This will add (lam_0, lam_1, ..., lam_n)
        productions for each argument type/arity combination, as well as
        ($i_j) productions for each variable de bruijn index i and type j.

        :param max_type_depth: The maximum depth of types to generate.
        """
        self.lambda_parameters = dict(max_type_depth=max_type_depth)

    def extra_productions(
        self, symbol: str, productions: List[Production], stable: bool = True
    ):
        """
        Add custom productions to the DSL. These are added as-is without
        type expansion. If stable is True, these productions will not be
        reindexed during pruning.

        :param symbol: The symbol group name for these productions (e.g., "<shield>").
        :param productions: The list of productions to add.
        :param stable: If True, these productions will not be reindexed during pruning.
        """
        self._extra_productions.append((symbol, productions, stable))

    def concrete(self, symbol: str, type_str: str, semantics: object):
        """
        Deprecated alias of :py:meth:`production`.
        """
        warnings.warn(
            "The method concrete is deprecated. Use production instead.",
            DeprecationWarning,
        )
        self.production(symbol, type_str, semantics, {})

    def parameterized(
        self,
        symbol: str,
        type_str: str,
        semantics: object,
        parameters: Dict[str, Callable[[], object]],
    ):
        """
        Deprecated alias of :py:meth:`production`.
        """
        warnings.warn(
            "The method parameterized is deprecated. Use production instead.",
            DeprecationWarning,
        )
        self.production(symbol, type_str, semantics, parameters)

    def production(
        self,
        symbol: str,
        type_str: str,
        semantics: object,
        parameters: Dict[str, Callable[[], object]] = None,
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
        if parameters is None:
            parameters = {}
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
        self, universe, production_constructor, symbol, sig, *args
    ):
        type_atoms, type_constructors = universe
        sigs = sorted(
            set(
                _signature_expansions(
                    sig,
                    type_atoms,
                    type_constructors,
                    max_expansion_steps=self.max_expansion_steps,
                    max_overall_depth=self.max_overall_depth,
                )
            ),
            key=str,
        )
        assert len(sigs) > 0, f"No expansions within depth/step bounds for {symbol}"

        prods = [
            production_constructor(
                symbol, FunctionTypeSignature.from_type(expansion), *args
            )
            for expansion in sigs
        ]

        return {symbol: Production.reindex(prods)}

    def _expansions_for_all_productions(self, universe, production_constructor, args):
        result = {}
        for arg in args:
            for_prod = self._expansions_for_single_production(
                universe, production_constructor, *arg
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

    def _build_lambda_productions(
        self, known_types, sym_to_productions, stable_symbols, needed_input_types
    ):
        """Build lambda and variable productions from discovered arrow types."""
        if needed_input_types is not None:
            expanded = sorted(
                [
                    ArrowType(inp, AtomicType("output_type"))
                    for inp in needed_input_types
                ],
                key=str,
            )
        else:
            # No pruning target: enumerate all possible lambda types.
            types, constructors_lambda = _type_universe(
                known_types,
                no_zeroadic=self._no_zeroadic,
            )
            top_levels = types + [
                constructor(
                    *[TypeVariable.fresh() for _ in range(arity)],
                )
                for arity, constructor in constructors_lambda
            ]
            top_levels = [
                x.with_output_type(AtomicType("output_type"))
                for x in top_levels
                if isinstance(x, ArrowType)
            ]
            top_levels = sorted(set(top_levels), key=str)
            effective_type_depth = min(
                self.lambda_parameters["max_type_depth"],
                self.max_overall_depth,
            )
            expanded = []
            for top_level in top_levels:
                expanded += type_expansions(
                    top_level,
                    types,
                    constructors_lambda,
                    max_expansion_steps=self.max_expansion_steps,
                    max_overall_depth=effective_type_depth,
                )
            expanded = sorted(set(expanded), key=str)

        sym_to_productions["<lambda>"] = [
            LambdaProduction(i, LambdaTypeSignature(x.input_type))
            for i, x in enumerate(expanded)
        ]
        variable_types = sorted(
            {inp for at in expanded for inp in at.input_type},
            key=str,
        )
        sym_to_productions["<variable>"] = [
            VariableProduction(
                type_id,
                VariableTypeSignature(variable_type, index_in_env),
            )
            for type_id, variable_type in enumerate(variable_types)
            for index_in_env in range(self.max_env_depth)
        ]
        # don't prune and reindex variables
        stable_symbols.add("<variable>")

    def finalize(self) -> DSL:
        """
        Produce the DSL from this factory. This will generate all productions and
        potentially raise errors if there were issues with the way the DSL was
        constructed.
        """

        known_types = (
            [x.astype() for x in self._signatures]
            + self._known_types
            + (self.target_types if self.target_types is not None else [])
        )

        universe = _type_universe(known_types, no_zeroadic=self._no_zeroadic)

        sym_to_productions: Dict[str, List[Production]] = {}
        sym_to_productions.update(
            self._expansions_for_all_productions(
                universe, ParameterizedProduction.of, self._parameterized_productions
            )
        )

        stable_symbols = set()
        needed_input_types = None
        constructible_syms = None

        if self.prune and self.target_types is not None:
            # Demand-driven: BFS from target types + constructibility fixed
            # point discovers which productions are reachable and which
            # ArrowTypes need lambda productions — replacing generate-then-prune.
            all_prods = [p for ps in sym_to_productions.values() for p in ps]
            for _, prods, _ in self._extra_productions:
                all_prods.extend(prods)

            max_lambda_depth = None
            if self.lambda_parameters is not None:
                max_lambda_depth = min(
                    self.lambda_parameters["max_type_depth"],
                    self.max_overall_depth,
                )

            constructible_syms, needed_input_types = _discover_reachable(
                all_prods,
                self.target_types,
                self.max_overall_depth,
                self.max_env_depth,
                max_lambda_depth=max_lambda_depth,
            )

            # Filter base productions to constructible only
            sym_to_productions = _filter_to_constructible(
                sym_to_productions,
                constructible_syms,
                stable_symbols,
                self.tolerate_pruning_entire_productions,
            )

        if self.lambda_parameters is not None:
            self._build_lambda_productions(
                known_types,
                sym_to_productions,
                stable_symbols,
                needed_input_types,
            )

        for symbol, prods, stable in self._extra_productions:
            if constructible_syms is not None:
                prods = [p for p in prods if p.symbol() in constructible_syms]
            sym_to_productions[symbol] = prods
            if stable:
                stable_symbols.add(symbol)

        if self.prune and self.prune_variables:
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


def _discover_reachable(
    all_productions,
    target_types,
    type_depth_limit,
    env_depth_limit,
    *,
    max_lambda_depth=None,
):
    """BFS from *target_types* through productions, then bottom-up constructibility.

    Combines top-down reachability with bottom-up constructibility in one pass,
    replacing the old pattern of generate-everything-then-prune.

    Returns ``(constructible_symbols, needed_input_types)`` where
    *needed_input_types* contains the input_type tuples of reachable ArrowTypes
    (used to create lambda productions), and *constructible_symbols* is the set
    of production symbol strings that survive constructibility analysis.
    """
    from ..types.type_with_environment import (
        PermissiveEnvironmment,
        TypeWithEnvironment,
    )

    needed_input_types = set()
    worklist = [TypeWithEnvironment(t, PermissiveEnvironmment()) for t in target_types]
    visited = set()
    # rules: bare Type -> [(symbol, [child Type, ...])]
    rules = {}
    while worklist:
        twe = worklist.pop()
        if (
            twe.typ.depth > type_depth_limit
            or len(twe.env) > env_depth_limit
            or twe in visited
        ):
            continue
        visited.add(twe)
        typ = twe.typ
        if typ not in rules:
            rules[typ] = []
        for prod in all_productions:
            arg_types = prod.type_signature().unify_return(twe)
            if arg_types is not None:
                rules[typ].append((prod.symbol(), [a.typ for a in arg_types]))
                worklist.extend(arg_types)
        if (
            max_lambda_depth is not None
            and isinstance(typ, ArrowType)
            and typ.depth < max_lambda_depth
        ):
            needed_input_types.add(typ.input_type)
            # Lambda rule: the body type must be constructible
            rules[typ].append(("<lambda>", [typ.output_type]))
            worklist.append(
                TypeWithEnvironment(
                    typ.output_type,
                    twe.env.child(*typ.input_type),
                )
            )
            # Variable rules: leaves (always constructible within a lambda)
            for inp in typ.input_type:
                if inp not in rules:
                    rules[inp] = []
                rules[inp].append(("<variable>", []))

    return _constructible_symbols(rules), needed_input_types


def _constructible_symbols(rules):
    """Bottom-up fixed point: find symbols whose arguments are all constructible."""
    constructible = set()
    changed = True
    while changed:
        changed = False
        for out_t, type_rules in rules.items():
            if out_t in constructible:
                continue
            for _, in_types in type_rules:
                if all(t in constructible for t in in_types):
                    constructible.add(out_t)
                    changed = True
                    break
    return {
        sym
        for type_rules in rules.values()
        for sym, in_types in type_rules
        if all(t in constructible for t in in_types)
    }


def _filter_to_constructible(
    sym_to_productions,
    constructible_syms,
    stable_symbols,
    tolerate_pruning_entire_productions,
):
    """Keep only productions whose symbols are in *constructible_syms*."""
    new_sym_to_productions = {}
    for original_symbol, prods in sym_to_productions.items():
        new_sym_to_productions[original_symbol] = [
            x for x in prods if x.symbol() in constructible_syms
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
