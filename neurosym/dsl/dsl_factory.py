import copy
import warnings
from typing import Callable, Dict, List, Tuple

import numpy as np

from ..types.type import ArrowType, AtomicType, Type, TypeVariable, UnificationError
from ..types.type_signature import (
    FunctionTypeSignature,
    LambdaTypeSignature,
    VariableTypeSignature,
    _signature_expansions,
    _type_universe,
    type_expansions,
)
from ..types.type_string_repr import TypeDefiner
from ..utils.documentation import internal_only
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
        self, type_atoms, type_constructors, production_constructor, symbol, sig, *args
    ):
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

    def _expansions_for_all_productions(
        self, type_atoms, type_constructors, production_constructor, args
    ):
        result = {}
        for arg in args:
            for_prod = self._expansions_for_single_production(
                type_atoms, type_constructors, production_constructor, *arg
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

        known_types = (
            [x.astype() for x in self._signatures]
            + self._known_types
            + (self.target_types if self.target_types is not None else [])
        )

        universe = _type_universe(known_types, no_zeroadic=self._no_zeroadic)

        has_lambdas = self.lambda_parameters is not None

        if self.prune:
            assert self.target_types is not None
            sym_to_productions = self._finalize_with_pruning(
                has_lambdas,
            )
        else:
            sym_to_productions = self._finalize_without_pruning(
                universe,
                has_lambdas,
                known_types,
            )

        if "<variable>" in sym_to_productions:
            sym_to_productions["<variable>"] = _clean_variables(
                sym_to_productions["<variable>"]
            )
        return _make_dsl(
            sym_to_productions,
            copy.copy(self.target_types),
            self.max_overall_depth,
            self.max_env_depth,
        )

    def _finalize_without_pruning(self, universe, has_lambdas, known_types):
        """Original expansion path: expand everything, no pruning."""
        sym_to_productions: Dict[str, List[Production]] = {}
        sym_to_productions.update(
            self._expansions_for_all_productions(
                *universe, ParameterizedProduction.of, self._parameterized_productions
            )
        )

        if has_lambdas:
            self._add_all_lambda_variable_productions(
                sym_to_productions,
                known_types,
            )

        for symbol, prods, _stable in self._extra_productions:
            sym_to_productions[symbol] = prods

        return sym_to_productions

    def _finalize_with_pruning(self, has_lambdas):
        """Pruning path using constructibility analysis."""
        # pylint: disable=too-many-branches,too-many-nested-blocks,too-many-statements
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
            sigs_only, has_lambdas, self.max_overall_depth, self.target_types
        )

        # Top-down: find reachable productions and lambdas
        reachable_prods, reachable_lambdas = reachable_symbols(
            named_sigs,
            constructible,
            self.target_types,
            has_lambdas,
            self.max_overall_depth,
        )

        # Group reachable substitutions by symbol, dedup by concrete type
        reachable_by_sym = {}
        for sym, subst_frozen in reachable_prods:
            reachable_by_sym.setdefault(sym, set()).add(subst_frozen)

        # Build concrete Production objects, preserving declaration order
        sym_to_productions: Dict[str, List[Production]] = {}
        for sym, sig, semantics, parameters in self._parameterized_productions:
            subst_set = reachable_by_sym.get(sym, set())
            if not subst_set:
                if not self.tolerate_pruning_entire_productions:
                    raise TypeError(
                        f"All productions for {sym} were pruned. "
                        f"Check that the target types are correct."
                    )
                sym_to_productions[sym] = []
                continue

            # Shared type variables (in both args and return) should be preserved
            arg_vars = {v for a in sig.arguments for v in a.get_type_vars()}
            ret_vars = set(sig.return_type.get_type_vars())
            shared_vars = arg_vars & ret_vars

            prods = {}
            for subst_frozen in subst_set:
                subst = dict(subst_frozen)
                filtered_subst = {
                    k: v for k, v in subst.items() if k not in shared_vars
                }
                concrete_type = sig.astype().subst_type_vars(filtered_subst)
                if concrete_type.depth > self.max_overall_depth:
                    continue
                if set(concrete_type.get_type_vars()) - shared_vars:
                    continue
                key = str(concrete_type)
                if key not in prods:
                    concrete_sig = FunctionTypeSignature.from_type(concrete_type)
                    prods[key] = ParameterizedProduction.of(
                        sym, concrete_sig, semantics, parameters
                    )
            sym_to_productions[sym] = Production.reindex(
                sorted(
                    prods.values(),
                    key=lambda p: str(p.type_signature().astype()),
                )
            )

        # Add lambda and variable productions from reachable lambdas
        if has_lambdas and reachable_lambdas:
            # Filter lambdas by max_type_depth: the arrow type with a
            # dummy output must fit within the lambda depth bound
            max_lambda_depth = self.lambda_parameters.get(
                "max_type_depth", self.max_overall_depth
            )
            reachable_lambdas = {
                inp
                for inp in reachable_lambdas
                if ArrowType(inp, AtomicType("_")).depth < max_lambda_depth
            }
            # Filter lambdas whose input types are never consumed by any
            # production, even transitively through lambda nesting.  A type
            # is "consumed" if it appears as an argument to some production,
            # or as an input element of an arrow type that is consumed.
            consumed_types = {
                arg
                for prods in sym_to_productions.values()
                for prod in prods
                for arg in prod.type_signature().arguments
            }
            if not any(t.has_type_vars() for t in consumed_types):
                # Expand consumed set: arrow types consume their input types
                changed = True
                while changed:
                    changed = False
                    for t in list(consumed_types):
                        if isinstance(t, ArrowType):
                            for inp_t in t.input_type:
                                if inp_t not in consumed_types:
                                    consumed_types.add(inp_t)
                                    changed = True
                reachable_lambdas = {
                    inp
                    for inp in reachable_lambdas
                    if any(t in consumed_types for t in inp)
                }
            lambda_input_types = sorted(reachable_lambdas, key=str)
            sym_to_productions["<lambda>"] = Production.reindex(
                [
                    LambdaProduction(i, LambdaTypeSignature(input_types))
                    for i, input_types in enumerate(lambda_input_types)
                ]
            )

            variable_types = sorted(
                {t for input_types in reachable_lambdas for t in input_types},
                key=str,
            )
            sym_to_productions["<variable>"] = [
                VariableProduction(
                    type_id, VariableTypeSignature(variable_type, index_in_env)
                )
                for type_id, variable_type in enumerate(variable_types)
                for index_in_env in range(self.max_env_depth)
            ]

        stable_symbols = set()
        for symbol, prods, stable in self._extra_productions:
            sym_to_productions[symbol] = prods
            if stable:
                stable_symbols.add(symbol)

        # Prune unreachable variable/shield productions.
        # Enumerate all (type, index) pairs reachable by nesting reachable lambdas.
        if self.prune_variables and "<variable>" in sym_to_productions:
            reachable_var_slots = set()
            if reachable_lambdas:
                lambda_inputs = sorted(reachable_lambdas, key=str)
                # BFS over lambda nestings. Each lambda can be nested multiple
                # times (e.g., lam(call(lam(...), x)) nests two copies).
                seen = {()}
                frontier = [()]
                while frontier:
                    env = frontier.pop()
                    for inp in lambda_inputs:
                        new_env = tuple(reversed(inp)) + env
                        if len(new_env) > self.max_env_depth:
                            continue
                        if new_env not in seen:
                            seen.add(new_env)
                            frontier.append(new_env)
                for env in seen:
                    for idx, typ in enumerate(env):
                        reachable_var_slots.add((typ, idx))
            for symbol in list(sym_to_productions):
                sig_list = sym_to_productions[symbol]
                filtered = []
                for prod in sig_list:
                    ts = prod.type_signature()
                    if hasattr(ts, "index_in_env") and hasattr(ts, "variable_type"):
                        if (
                            ts.variable_type,
                            ts.index_in_env,
                        ) not in reachable_var_slots:
                            continue
                    elif hasattr(ts, "index_in_env"):
                        # Shield: keep if any type at this index is reachable
                        if not any(
                            idx == ts.index_in_env for _, idx in reachable_var_slots
                        ):
                            continue
                    filtered.append(prod)
                if symbol not in stable_symbols:
                    filtered = Production.reindex(filtered)
                sym_to_productions[symbol] = filtered

        return sym_to_productions

    def _add_all_lambda_variable_productions(self, sym_to_productions, known_types):
        """Add all possible lambda and variable productions (old expansion path)."""
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
        expanded = []
        for top_level in top_levels:
            expanded += type_expansions(
                top_level,
                types,
                constructors_lambda,
                max_expansion_steps=self.max_expansion_steps,
                max_overall_depth=self.lambda_parameters["max_type_depth"],
            )
        expanded = sorted(set(expanded), key=str)
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
            for index_in_env in range(self.max_env_depth)
        ]


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


class _ConstructibilityChecker:
    """Shared logic for checking type constructibility with environment support.

    Tracks a dict mapping environments (frozensets of Types) to sets of types
    nontrivially constructible in that environment. Provides methods for
    checking constructibility and finding type variable bindings.
    """

    def __init__(self, has_lambdas, register_envs=False):
        self.has_lambdas = has_lambdas
        self.register_envs = register_envs
        self.constructible = {frozenset(): set()}

    def all_constructible_in(self, env):
        """All types constructible in env (including env members and sub-envs)."""
        result = set(env)
        for tracked_env, types in self.constructible.items():
            if tracked_env <= env:
                result |= types
        return result

    def is_constructible(self, t, env=frozenset()):
        """Check if type t is constructible in the given environment."""
        if t in env:
            return True
        for tracked_env, types in self.constructible.items():
            if tracked_env <= env and t in types:
                return True
        if self.has_lambdas and isinstance(t, ArrowType):
            new_env = env | frozenset(t.input_type)
            if self.register_envs and new_env not in self.constructible:
                self.constructible[new_env] = set()
            return self.is_constructible(t.output_type, new_env)
        return False

    def bindings_for(self, pattern, subst, env=frozenset()):
        """Yield extended substitutions that make pattern constructible in env."""
        resolved = pattern.subst_type_vars(subst)
        if not resolved.get_type_vars():
            if self.is_constructible(resolved, env):
                yield subst
            return
        for t in self.all_constructible_in(env):
            try:
                new_bindings = resolved.unify(t)
            except UnificationError:
                continue
            merged = _merge_subst(subst, new_bindings)
            if merged is not None:
                yield merged
        if self.has_lambdas and isinstance(resolved, ArrowType):
            yield from self.bindings_for(
                resolved.output_type,
                subst,
                env | frozenset(resolved.input_type),
            )

    def find_valid_substs(self, sig, env=frozenset()):
        """Yield substitutions that make all arguments of sig constructible in env."""
        return self.find_valid_substs_with_initial(sig, {}, env)

    def find_valid_substs_with_initial(self, sig, initial_subst, env=frozenset()):
        """Like find_valid_substs but starting from an initial substitution."""
        substs = [initial_subst]
        for arg in sig.arguments:
            next_substs = []
            for subst in substs:
                next_substs.extend(self.bindings_for(arg, subst, env))
            substs = next_substs
            if not substs:
                break
        return substs


def _merge_subst(base, extension):
    """Merge two substitutions, return None if inconsistent."""
    merged = dict(base)
    for k, v in extension.items():
        if k in merged:
            if merged[k] != v:
                return None
        else:
            merged[k] = v
    return merged


@internal_only
def directly_constructible_types(signatures, has_lambdas, max_depth, target_types=None):
    """
    Compute the set of constructible types per environment via a bottom-up fixed point,
    working directly from raw production signatures (which may contain type variables).

    A type is constructible in environment E if:
    - It is constructible in a sub-environment of E (including the empty env), OR
    - It is a member of E (a variable), OR
    - It is an arrow type ``(A1, ..., An) -> B`` where ``B`` is constructible in
      ``E ∪ {A1, ..., An}`` (when has_lambdas is True), OR
    - Some production can output it with all inputs constructible in E.

    Returns a dict mapping each environment (frozenset of Types) to the set of types
    nontrivially constructible in that environment — excluding types that are members
    of the environment or constructible in a strict sub-environment.
    The empty-env entry (``frozenset()``) holds the directly constructible types.
    """
    # pylint: disable=too-many-branches
    checker = _ConstructibilityChecker(has_lambdas, register_envs=True)
    constructible = checker.constructible

    # Seed envs from arrow-typed targets so their bodies get explored
    if has_lambdas and target_types:
        for t in target_types:
            if isinstance(t, ArrowType):
                env = frozenset(t.input_type)
                if env not in constructible:
                    constructible[env] = set()

    while True:
        prev_env_count = len(constructible)
        done = True
        for env in list(constructible.keys()):
            for sig in signatures:
                for subst in checker.find_valid_substs(sig, env):
                    out_t = sig.return_type.subst_type_vars(subst)
                    if out_t.get_type_vars() or out_t.depth > max_depth:
                        continue
                    if not checker.is_constructible(out_t, env):
                        constructible[env].add(out_t)
                        done = False
        if len(constructible) != prev_env_count:
            done = False
        if done:
            break

    # Clean up: remove types constructible in a strict sub-env,
    # and remove empty env entries (except the root empty env)
    for env in constructible:
        for sub_env in constructible:
            if sub_env < env:
                constructible[env] -= constructible[sub_env]

    return {
        env: types
        for env, types in constructible.items()
        if types or env == frozenset()
    }


def _add_targets_needed(t, env, frontier, lambdas, has_lambdas):
    """Add (type, env) pairs to frontier for constructing t in env.

    For arrow types with lambdas, records the lambda and recurses on the body.
    """
    frontier.append((t, env))
    if has_lambdas and isinstance(t, ArrowType):
        lambdas.add(t.input_type)
        _add_targets_needed(
            t.output_type,
            env | frozenset(t.input_type),
            frontier,
            lambdas,
            has_lambdas,
        )


@internal_only
def reachable_symbols(signatures, constructible, target_types, has_lambdas, max_depth):
    """
    Top-down search from target types through signatures, collecting concrete
    production instantiations and lambda argument types that are reachable.

    Starting from ``target_types``, find all signatures whose return type
    unifies with a needed type and whose arguments are all constructible
    (according to ``constructible``). Record the symbol with its type variable
    bindings, and recurse on the argument types.

    :param signatures: List of (symbol, FunctionTypeSignature) pairs.
    :param constructible: Dict from env (frozenset) to set of nontrivially
        constructible types, as returned by ``directly_constructible_types``.
    :param target_types: List of Type objects to start the search from.
    :param has_lambdas: Whether lambdas are enabled.
    :param max_depth: Maximum type depth.

    :return: A tuple of ``(productions, lambdas)`` where:
        - ``productions`` is a set of ``(symbol, subst)`` pairs, where ``subst``
          is a frozenset of ``(var_name, Type)`` items representing the type
          variable bindings for that instantiation.
        - ``lambdas`` is a set of tuples of Types, each representing the input
          types of a lambda that is needed (i.e., the argument types of an arrow
          type that is constructed via lambda).
    """
    # pylint: disable=too-many-branches,too-many-nested-blocks
    checker = _ConstructibilityChecker(has_lambdas)
    checker.constructible = constructible

    productions = set()
    lambdas = set()
    visited = set()

    def _enqueue(t, env):
        """Add (type, env) to frontier if not already visited and constructible."""
        if (t, env) in visited or t.depth > max_depth:
            return
        if not checker.is_constructible(t, env):
            return
        visited.add((t, env))
        frontier.append((t, env))

    def _enqueue_with_lambda(t, env):
        """Enqueue t. If arrow type with lambdas, record the lambda and enqueue body."""
        _enqueue(t, env)
        if has_lambdas and isinstance(t, ArrowType):
            lambdas.add(t.input_type)
            _enqueue_with_lambda(t.output_type, env | frozenset(t.input_type))

    frontier = []
    for t in target_types:
        _enqueue_with_lambda(t, frozenset())

    while frontier:
        target, env = frontier.pop()

        # Arrow targets can be produced by lambda (already recorded above)
        # but also by productions — so we search for both.

        for sym, sig in signatures:
            try:
                ret_subst = sig.return_type.unify(target)
            except UnificationError:
                continue

            for subst in checker.find_valid_substs_with_initial(sig, ret_subst, env):
                is_new = (sym, frozenset(subst.items())) not in productions
                productions.add((sym, frozenset(subst.items())))
                if is_new:
                    for arg in sig.arguments:
                        resolved_arg = arg.subst_type_vars(subst)
                        if not resolved_arg.get_type_vars():
                            # Record lambdas only for direct production arguments
                            _enqueue_with_lambda(resolved_arg, env)

    return productions, lambdas
