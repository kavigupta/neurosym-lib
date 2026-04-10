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
        Add lambda productions to the DSL. This will add polymorphic lambda
        productions for each reachable arity, as well as polymorphic variable
        productions ($0, $1, ...) for each de bruijn index.

        :param max_type_depth: Maximum depth of types to explore when determining
            which types are constructible inside lambda bodies. Controls how deeply
            nested list/arrow types get expanded for productions with type variables.
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
        self.target_types = [self.t(x) for x in target_types]
        self.prune_variables = prune_variables
        self.tolerate_pruning_entire_productions = tolerate_pruning_entire_productions

    def finalize(self) -> DSL:
        """
        Produce the DSL from this factory. This will generate all productions and
        potentially raise errors if there were issues with the way the DSL was
        constructed.
        """

        if self.target_types is None:
            raise TypeError("prune_to() must be called before finalize()")

        has_lambdas = self.lambda_parameters is not None
        named_sigs = [(sym, sig) for sym, sig, _, _ in self._parameterized_productions]

        # Check for duplicate declarations
        seen = {}
        for sym, sig in named_sigs:
            if sym in seen:
                if seen[sym] != sig:
                    raise ValueError(f"Duplicate declarations for production: {sym}")
            else:
                seen[sym] = sig

        # Bottom-up: compute constructible types
        sigs_only = [sig for _, sig in named_sigs]
        constructible = directly_constructible_types(
            sigs_only,
            has_lambdas,
            self.max_overall_depth,
            self.target_types,
        )

        # Top-down: find reachable productions and lambda arrow types
        max_lambda_depth = (
            self.lambda_parameters.get("max_type_depth", self.max_overall_depth)
            if has_lambdas
            else self.max_overall_depth
        )
        reachable_prods, lambda_arrows = reachable_symbols(
            named_sigs,
            constructible,
            self.target_types,
            has_lambdas,
            self.max_overall_depth,
            max_lambda_depth,
        )

        sym_to_productions = self._build_concrete_productions(reachable_prods)

        reachable_lambda_arities = set()
        if has_lambdas and lambda_arrows:
            reachable_lambda_arities = _useful_arities(
                lambda_arrows, sym_to_productions
            )

        if reachable_lambda_arities:
            _add_lambda_variable_productions(
                sym_to_productions, reachable_lambda_arities, self.max_env_depth
            )

        reachable_indices = (
            set(range(self.max_env_depth)) if reachable_lambda_arities else set()
        )
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

        return DSL(
            [prod for prods in sym_to_productions.values() for prod in prods],
            copy.copy(self.target_types),
            self.max_overall_depth,
            max_env_depth=self.max_env_depth,
        )

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


def _useful_arities(lambda_arrows, sym_to_productions):
    """Return the set of lambda arities that are actually usable.

    An arity is useful if some reachable arrow type of that arity has all
    its input types consumed by some production. This prevents creating
    lambdas for arrow types like ``{f, 14} -> {f, 1}`` when no production
    takes ``{f, 14}`` as an argument.

    Skips the consumed-types check if any production has polymorphic
    arguments (which could consume any type).
    """
    consumed_types = {
        arg
        for prods in sym_to_productions.values()
        for prod in prods
        for arg in prod.type_signature().arguments
    }
    if not consumed_types or any(t.has_type_vars() for t in consumed_types):
        return {len(a.input_type) for a in lambda_arrows}
    # Transitively: arrow types consume their input element types.
    changed = True
    while changed:
        changed = False
        for t in list(consumed_types):
            if isinstance(t, ArrowType):
                for inp_t in t.input_type:
                    if inp_t not in consumed_types:
                        consumed_types.add(inp_t)
                        changed = True
    return {
        len(a.input_type)
        for a in lambda_arrows
        if all(t in consumed_types for t in a.input_type)
    }


def _add_lambda_variable_productions(
    sym_to_productions, reachable_lambda_arities, max_env_depth
):
    """Add lambda and variable productions for the given reachable lambda arities."""
    sorted_arities = sorted(reachable_lambda_arities)
    sym_to_productions["<lambda>"] = Production.reindex(
        [
            LambdaProduction(i, LambdaTypeSignature(arity))
            for i, arity in enumerate(sorted_arities)
        ]
    )
    sym_to_productions["<variable>"] = [
        VariableProduction(VariableTypeSignature(idx)) for idx in range(max_env_depth)
    ]
