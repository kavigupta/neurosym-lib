from typing import Callable, Dict, List

import numpy as np

from neurosym.dsl.dsl import ROOT_SYMBOL
from neurosym.dsl.production import Production
from neurosym.types.type import Type, UnificationError
from neurosym.types.type_signature import FunctionTypeSignature, resolve_type
from neurosym.types.type_with_environment import StrictEnvironment, TypeWithEnvironment

from .preorder_mask import PreorderMask


class TypePreorderMask(PreorderMask):
    """
    Masks out productions that would lead to ill-typed programs. This mask is
    computed by traversing the tree in a preorder fashion and checking the types
    of the children of the current node.

    Tracks type variable bindings across siblings: when a production resolves
    a type variable (e.g., choosing ``one :: () -> int`` for a hole of type ``#a``),
    the binding ``#a = int`` is propagated to sibling holes that share the same
    variable.

    This mask can be cached by the tree distribution to speed up enumeration, the
    cache key is the unique hash of the resolved ``TypeWithEnvironment``.
    """

    def __init__(self, tree_dist, dsl):
        super().__init__(tree_dist)
        self.dsl = dsl
        assert len(self.dsl.valid_root_types) == 1, "Only one root type is supported"
        self.root_type = self.dsl.valid_root_types[0]

        # stack of list of type_with_environment objects
        # each list represents the types of the children of the current node
        self.type_stack: List[List[TypeWithEnvironment]] = []

        # global type variable bindings, accumulated during tree traversal.
        # keys are fresh variable names (from alpha-renaming); values are Types.
        self.bindings: Dict[str, Type] = {}

    def _resolve_twe(self, twe: TypeWithEnvironment) -> TypeWithEnvironment:
        """Resolve type variables in a TypeWithEnvironment using current bindings."""
        resolved = resolve_type(twe.typ, self.bindings)
        return TypeWithEnvironment(resolved, twe.env)

    def valid_productions(self, twe: TypeWithEnvironment) -> List[Production]:
        """
        Compute the set of valid productions for a given type with environment.
        Can be overridden by subclasses to implement different type masking strategies.

        :param twe: The type to compute valid symbols for. Should already be resolved.
        :return: A list of valid productions.
        """
        return [prod for prod, _ in self.dsl.productions_for_type(twe)]

    def compute_mask(self, position, symbols):
        twe = self.type_stack[-1][position]
        resolved_twe = self._resolve_twe(twe)
        valid_prods = self.valid_productions(resolved_twe)
        valid_indices = {
            self.tree_dist.symbol_to_index[prod.symbol()] for prod in valid_prods
        }
        return np.where(
            (
                np.isin(symbols, list(valid_indices))
                if len(valid_indices) > 0
                else np.zeros(len(symbols), dtype=bool)
            ),
            0.0,
            -np.inf,
        )

    def on_entry(self, position, symbol) -> Callable[[], None]:
        symbol_str, arity = self.tree_dist.symbols[symbol]
        if symbol_str == ROOT_SYMBOL:
            self.type_stack.append(
                [TypeWithEnvironment(self.root_type, StrictEnvironment.empty())]
            )
            return self.type_stack.pop

        parent_twe = self.type_stack[-1][position]
        resolved_parent = self._resolve_twe(parent_twe)

        production = self.dsl.get_production(symbol_str)
        sig = production.type_signature()

        new_binding_keys = []

        if isinstance(sig, FunctionTypeSignature):
            # Alpha-rename to avoid name collisions between production usages
            fresh_sig, _ = sig.alpha_rename()

            # Unify the resolved parent type with the fresh return type
            try:
                mapping = resolved_parent.typ.unify(fresh_sig.return_type)
            except UnificationError as exc:
                raise ValueError(
                    f"Type mismatch: {production} cannot produce {resolved_parent}"
                ) from exc

            # Record new bindings
            for k, v in mapping.items():
                if k not in self.bindings:
                    self.bindings[k] = v
                    new_binding_keys.append(k)

            # Compute children types using the mapping
            children_types = [
                TypeWithEnvironment(
                    t.subst_type_vars(mapping), resolved_parent.env
                )
                for t in fresh_sig.arguments
            ]
        else:
            # Lambda/Variable signatures: already concrete, no alpha-renaming needed
            children_types = sig.unify_return(resolved_parent)

        if children_types is None:
            raise ValueError(
                f"Type mismatch in production {production} with parent type {resolved_parent}"
            )
        assert len(children_types) == arity
        self.type_stack.append(children_types)

        def undo():
            self.type_stack.pop()
            for k in new_binding_keys:
                del self.bindings[k]

        return undo

    def on_exit(self, position, symbol) -> Callable[[], None]:
        del position
        del symbol
        last = self.type_stack.pop()
        return lambda: self.type_stack.append(last)

    def cache_key(self, parents):
        position = parents[-1][1]
        twe = self.type_stack[-1][position]
        resolved_twe = self._resolve_twe(twe)
        return resolved_twe.unique_hash
