import copy
import warnings
from typing import Callable, Dict, List, Tuple

from ..types.type import ArrowType, Type
from ..types.type_signature import LambdaTypeSignature, VariableTypeSignature
from ..types.type_string_repr import TypeDefiner
from .constructibility import directly_constructible_types, reachable_symbols
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

    def __init__(self, max_env_depth=4, max_overall_depth=6, **env):
        self.t = TypeDefiner(**env)
        self._parameterized_productions = []
        self.lambda_parameters = None
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

    def finalize(self) -> DSL:
        """
        Produce the DSL from this factory. This will generate all productions and
        potentially raise errors if there were issues with the way the DSL was
        constructed.
        """

        if not self.prune:
            raise TypeError("prune_to() must be called before finalize()")

        has_lambdas = self.lambda_parameters is not None

        sym_to_productions = self._finalize_with_pruning(
            has_lambdas,
        )

        return _make_dsl(
            sym_to_productions,
            copy.copy(self.target_types),
            self.max_overall_depth,
            self.max_env_depth,
        )

    def _finalize_with_pruning(self, has_lambdas):
        """Pruning path using constructibility analysis."""
        named_sigs = [(sym, sig) for sym, sig, _, _ in self._parameterized_productions]

        # Check for duplicate declarations
        seen = {}
        for sym, sig in named_sigs:
            if sym in seen:
                if seen[sym] != sig:
                    raise ValueError(f"Duplicate declarations for production: {sym}")
            else:
                seen[sym] = sig

        if has_lambdas:
            max_lambda_depth = self.lambda_parameters.get(
                "max_type_depth", self.max_overall_depth
            )
        else:
            max_lambda_depth = self.max_overall_depth

        # Bottom-up: compute constructible types
        sigs_only = [sig for _, sig in named_sigs]
        constructible = directly_constructible_types(
            sigs_only,
            has_lambdas,
            self.max_overall_depth,
            self.target_types,
        )

        # Top-down: find reachable productions and lambdas
        reachable_prods, reachable_lambdas = reachable_symbols(
            named_sigs,
            constructible,
            self.target_types,
            has_lambdas,
            self.max_overall_depth,
            max_lambda_depth,
        )

        sym_to_productions = self._build_concrete_productions(reachable_prods)

        var_slots = set()
        if has_lambdas and reachable_lambdas:
            reachable_lambdas = _filter_useless_lambdas(
                reachable_lambdas, sym_to_productions
            )
            var_slots = _reachable_var_slots(reachable_lambdas, self.max_env_depth)
            _add_lambda_variable_productions(
                sym_to_productions, reachable_lambdas, var_slots
            )

        reachable_indices = {idx for _, idx in var_slots}
        for symbol, prods, stable in self._extra_productions:
            if self.prune_variables:
                prods = [
                    p
                    for p in prods
                    if (idx := p.type_signature().required_env_index()) is None
                    or idx in reachable_indices
                ]
                if not stable:
                    prods = Production.reindex(prods)
            sym_to_productions[symbol] = prods

        return sym_to_productions

    def _build_concrete_productions(self, prod_sigs):
        """Build concrete Production objects from {sym: [FunctionTypeSignature]}."""
        sym_to_productions: Dict[str, List[Production]] = {}
        for sym, _sig, semantics, parameters in self._parameterized_productions:
            concrete_sigs = prod_sigs.get(sym, [])
            if not concrete_sigs and not self.tolerate_pruning_entire_productions:
                raise TypeError(
                    f"All productions for {sym} were pruned. "
                    f"Check that the target types are correct."
                )
            sym_to_productions[sym] = Production.reindex(
                sorted(
                    [
                        ParameterizedProduction.of(sym, cs, semantics, parameters)
                        for cs in concrete_sigs
                    ],
                    key=lambda p: str(p.type_signature().astype()),
                )
            )
        return sym_to_productions


def _make_dsl(sym_to_productions, valid_root_types, max_type_depth, max_env_depth):
    return DSL(
        [prod for prods in sym_to_productions.values() for prod in prods],
        valid_root_types,
        max_type_depth,
        max_env_depth=max_env_depth,
    )


def _reachable_var_slots(reachable_lambdas, max_env_depth):
    """Compute reachable (type, env_index) pairs from lambda input types.

    Every type introduced by any lambda can in principle appear at any
    environment index. Slots that are unreachable due to arity constraints
    are harmless — they produce variable productions that never match
    during enumeration.
    """
    all_types = {t for inp in reachable_lambdas for t in inp}
    return {(t, i) for t in all_types for i in range(max_env_depth)}


def _add_lambda_variable_productions(sym_to_productions, reachable_lambdas, var_slots):
    """Add lambda and variable productions for the given reachable lambda types."""
    lambda_input_types = sorted(reachable_lambdas, key=str)
    sym_to_productions["<lambda>"] = Production.reindex(
        [
            LambdaProduction(i, LambdaTypeSignature(input_types))
            for i, input_types in enumerate(lambda_input_types)
        ]
    )
    variable_types = sorted({t for t, _ in var_slots}, key=str)
    type_to_idx = {t: i for i, t in enumerate(variable_types)}
    sym_to_productions["<variable>"] = [
        VariableProduction(type_to_idx[typ], VariableTypeSignature(typ, idx))
        for typ, idx in sorted(var_slots, key=lambda s: (s[1], str(s[0])))
    ]


def _filter_useless_lambdas(reachable_lambdas, sym_to_productions):
    """Filter lambdas whose input types are never consumed by any production.

    A type is "consumed" if it appears as an argument to some concrete
    production, or as an input element of an arrow type that is consumed
    (transitively). Lambdas that only introduce unconsumed types into the
    environment are useless — nothing in the DSL can use those variables.
    """
    consumed_types = {
        arg
        for prods in sym_to_productions.values()
        for prod in prods
        for arg in prod.type_signature().arguments
    }
    if not consumed_types:
        return reachable_lambdas
    if any(t.has_type_vars() for t in consumed_types):
        return reachable_lambdas
    # Expand: arrow types transitively consume their input element types
    changed = True
    while changed:
        changed = False
        for t in list(consumed_types):
            if isinstance(t, ArrowType):
                for inp_t in t.input_type:
                    if inp_t not in consumed_types:
                        consumed_types.add(inp_t)
                        changed = True
    return {inp for inp in reachable_lambdas if any(t in consumed_types for t in inp)}
