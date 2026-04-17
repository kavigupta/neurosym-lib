from typing import Callable, List

import numpy as np

from neurosym.dsl.dsl import ROOT_SYMBOL
from neurosym.dsl.production import Production, VariableProduction
from neurosym.types.type_with_environment import StrictEnvironment, TypeWithEnvironment

from .preorder_mask import PreorderMask


class TypePreorderMask(PreorderMask):
    """
    Masks out productions that would lead to ill-typed programs. This mask is
    computed by traversing the tree in a preorder fashion and checking the types
    of the children of the current node.

    This mask can be cached by the tree distribution to speed up enumeration, the
    cache key is the unique hash of the ``TypeWithEnvironment`` of the current node.
    """

    def __init__(self, tree_dist, dsl, dreamcoder_compat=False):
        super().__init__(tree_dist)
        self.dsl = dsl
        self.dreamcoder_compat = dreamcoder_compat
        self._validate_root_types()

        # stack of list of type_with_environment objects
        # each list represents the types of the children of the current node
        self.type_stack: List[List[TypeWithEnvironment]] = []

    def _validate_root_types(self):
        """
        Validate that the DSL's ``valid_root_types`` is compatible with this mask.

        By default the mask supports exactly one root type; subclasses that
        override :meth:`_root_type_for_entry` to supply the root type from
        some other source (e.g., per-task context) can relax this by
        overriding this method.
        """
        assert (
            self.dsl.valid_root_types is not None
            and len(self.dsl.valid_root_types) == 1
        ), "Only one root type is supported"

    def _root_type_for_entry(self):
        """
        Return the root type to use when entering the ROOT_SYMBOL position.

        Default is the DSL's single ``valid_root_types`` entry. Subclasses may
        override to supply a dynamically chosen (e.g., task-specific) type.
        """
        return self.dsl.valid_root_types[0]

    def valid_productions(self, twe: TypeWithEnvironment) -> List[Production]:
        """
        Compute the set of valid productions for a given type with environment.
        Can be overridden by subclasses to implement different type masking strategies.

        :param twe: The type to compute valid symbols for.
        :return: A list of valid productions.
        """
        return [prod for prod, _ in self.dsl.productions_for_type(twe)]

    def compute_mask(self, position, symbols):
        valid_prods = self.valid_productions(self.type_stack[-1][position])
        valid_indices = {
            self.tree_dist.symbol_to_index[prod.symbol()] for prod in valid_prods
        }
        mask = np.where(
            (
                np.isin(symbols, list(valid_indices))
                if len(valid_indices) > 0
                else np.zeros(len(symbols), dtype=bool)
            ),
            0.0,
            -np.inf,
        )
        if self.dreamcoder_compat:
            var_indices = {
                self.tree_dist.symbol_to_index[prod.symbol()]
                for prod in valid_prods
                if isinstance(prod, VariableProduction)
            }
            n_vars = len(var_indices)
            if n_vars > 1:
                var_adj = -np.log(n_vars)
                symbols_arr = np.asarray(symbols)
                for vi in var_indices:
                    mask[symbols_arr == vi] = var_adj
        return mask

    def on_entry(self, position, symbol) -> Callable[[], None]:
        symbol, arity = self.tree_dist.symbols[symbol]
        if symbol == ROOT_SYMBOL:
            self.type_stack.append(
                [
                    TypeWithEnvironment(
                        self._root_type_for_entry(), StrictEnvironment.empty()
                    )
                ]
            )
            return self.type_stack.pop
        parent_type = self.type_stack[-1][position]
        production = self.dsl.get_production(symbol)
        children_types = production.type_signature().unify_return(parent_type)
        if children_types is None:
            raise ValueError(
                f"Type mismatch in production {production} with parent type {parent_type}"
            )
        assert len(children_types) == arity
        self.type_stack.append(children_types)
        return self.type_stack.pop

    def on_exit(self, position, symbol) -> Callable[[], None]:
        del position
        del symbol
        last = self.type_stack.pop()
        return lambda: self.type_stack.append(last)

    def cache_key(self, parents):
        position = parents[-1][1]
        typ = self.type_stack[-1][position]
        return typ.unique_hash
