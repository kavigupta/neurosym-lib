from ..types.type import ArrowType, UnificationError
from ..types.type_signature import FunctionTypeSignature
from ..utils.documentation import internal_only


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


@internal_only
def reachable_symbols(
    signatures,
    constructible,
    target_types,
    has_lambdas,
    max_depth,
):
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

    :return: A dict mapping each symbol name to a list of
        ``FunctionTypeSignature`` objects. Type variables that appear in
        the return type are preserved as polymorphic; only input-only
        type variables are substituted with concrete types.
    """
    # pylint: disable=too-many-branches,too-many-nested-blocks
    checker = _ConstructibilityChecker(has_lambdas)
    checker.constructible = constructible

    # Precompute which type vars to preserve as polymorphic per signature.
    # Any var that appears in the return type stays polymorphic; only
    # input-only vars get substituted with concrete types.
    return_vars_by_sym = {}
    for sym, sig in signatures:
        return_vars_by_sym[sym] = set(sig.return_type.get_type_vars())

    seen_substs = set()
    prod_sigs = {}  # sym -> {type_str: FunctionTypeSignature}
    visited = set()

    def _enqueue(t, env):
        if (t, env) in visited or t.depth > max_depth:
            return
        if not checker.is_constructible(t, env):
            return
        visited.add((t, env))
        frontier.append((t, env))

    def _enqueue_with_lambda(t, env):
        _enqueue(t, env)
        if has_lambdas and isinstance(t, ArrowType) and t.depth < max_depth:
            _enqueue_with_lambda(t.output_type, env | frozenset(t.input_type))

    def _record(sym, sig, subst):
        """Record a concrete production, return True if new (for exploration)."""
        subst_key = (sym, frozenset(subst.items()))
        if subst_key in seen_substs:
            return False
        seen_substs.add(subst_key)

        return_vars = return_vars_by_sym[sym]
        filtered = {k: v for k, v in subst.items() if k not in return_vars}
        concrete_type = sig.astype().subst_type_vars(filtered)
        if concrete_type.depth > max_depth:
            return True
        if set(concrete_type.get_type_vars()) - return_vars:
            return True
        key = str(concrete_type)
        prod_sigs.setdefault(sym, {})[key] = FunctionTypeSignature.from_type(
            concrete_type
        )
        return True

    frontier = []
    for t in target_types:
        _enqueue_with_lambda(t, frozenset())

    while frontier:
        target, env = frontier.pop()

        for sym, sig in signatures:
            try:
                ret_subst = sig.return_type.unify(target)
            except UnificationError:
                continue

            for subst in checker.find_valid_substs_with_initial(sig, ret_subst, env):
                if _record(sym, sig, subst):
                    for arg in sig.arguments:
                        resolved_arg = arg.subst_type_vars(subst)
                        if not resolved_arg.get_type_vars():
                            _enqueue_with_lambda(resolved_arg, env)

    return {sym: list(sigs.values()) for sym, sigs in prod_sigs.items()}
