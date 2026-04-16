from dataclasses import dataclass
from typing import Callable, List, Tuple, Union

import numpy as np

from neurosym.dsl.dsl import ROOT_SYMBOL
from neurosym.dsl.production import Production, VariableProduction
from neurosym.types.type_with_environment import StrictEnvironment, TypeWithEnvironment

from .preorder_mask import PreorderMask


@dataclass(frozen=True)
class _UnionTypeWithEnvironment:
    """
    Represents a union of possible types at a position. Used when there are
    multiple valid root types: the direct child of ROOT_SYMBOL may satisfy
    any one of them.
    """

    types: Tuple[TypeWithEnvironment, ...]

    @property
    def unique_hash(self):
        return tuple(t.unique_hash for t in self.types)


# Type alias for what can live at a position in the type stack
_StackEntry = Union[TypeWithEnvironment, _UnionTypeWithEnvironment]


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
        assert (
            self.dsl.valid_root_types is not None
        ), "DSL must have valid_root_types set"

        # stack of list of type_with_environment objects
        # each list represents the types of the children of the current node
        self.type_stack: List[List[_StackEntry]] = []

    def valid_productions(self, twe: _StackEntry) -> List[Production]:
        """
        Compute the set of valid productions for a given type with environment.
        Can be overridden by subclasses to implement different type masking strategies.

        When ``twe`` is a :class:`_UnionTypeWithEnvironment`, returns the union of
        valid productions across all constituent types.

        :param twe: The type (or union of types) to compute valid symbols for.
        :return: A list of valid productions.
        """
        if isinstance(twe, _UnionTypeWithEnvironment):
            seen: set = set()
            result = []
            for t in twe.types:
                for prod, _ in self.dsl.productions_for_type(t):
                    if prod.symbol() not in seen:
                        seen.add(prod.symbol())
                        result.append(prod)
            return result
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
            root_twes = [
                TypeWithEnvironment(t, StrictEnvironment.empty())
                for t in self.dsl.valid_root_types
            ]
            if len(root_twes) == 1:
                entry: _StackEntry = root_twes[0]
            else:
                entry = _UnionTypeWithEnvironment(tuple(root_twes))
            self.type_stack.append([entry])
            return self.type_stack.pop
        parent_type = self.type_stack[-1][position]
        production = self.dsl.get_production(symbol)
        if isinstance(parent_type, _UnionTypeWithEnvironment):
            children_types = None
            for twe in parent_type.types:
                result = production.type_signature().unify_return(twe)
                if result is not None:
                    children_types = result
                    break
        else:
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
