import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from neurosym.dsl.dsl import DSL
from neurosym.program_dist.tree_distribution.preorder_mask.preorder_mask import (
    PreorderMask,
)
from neurosym.program_dist.tree_distribution.preorder_mask.undos import chain_undos
from neurosym.program_dist.tree_distribution.tree_distribution import TreeDistribution

from .extra_var import ExtraVar, canonicalized_python_name_leaf_regex
from .handler import DefaultHandler, HandlerPuller, default_handler
from .names import NAME_REGEX
from .special_case_symbol_predicate import NameEPredicate, SpecialCaseSymbolPredicate


@dataclass
class DefUseMaskConfiguration:
    """
    Configuration for the DefUseMask.
    """

    dfa: Dict
    node_hooks: Dict[str, HandlerPuller]

    def get_hook(self, symbol):
        prefixes = [x for x in self.node_hooks if symbol.startswith(x)]
        if not prefixes:
            return None
        assert len(prefixes) == 1, f"Multiple hooks found for {symbol}: {prefixes}"
        return self.node_hooks[prefixes[0]]

    def pull_handler(
        self,
        position: int,
        symbol: str,
        mask: "DefUseChainPreorderMask",
        defined_production_idxs: List[int],
        handler_fn=default_handler,
    ):
        hook = self.get_hook(symbol)
        if hook is None:
            return None
        return hook.pull_handler(
            position,
            symbol,
            mask,
            defined_production_idxs,
            self,
            handler_fn=handler_fn,
        )


class DefUseChainPreorderMask(PreorderMask):
    """
    Preorder mask that filters out symbols that are not defined at a given position.

    The idea is to have a stack of Handler objects, one for each node in the syntax tree.

    Args:
        tree_dist: The tree distribution that the mask is applied to.
        dsl: The domain-specific language that the mask is applied to.
        config: The configuration for the mask.
        special_case_predicate_fns: A list of functions that return special case predicates.
    """

    def __init__(
        self,
        tree_dist: TreeDistribution,
        dsl: DSL,
        config: DefUseMaskConfiguration,
        special_case_predicate_fns: Tuple[
            Callable[[TreeDistribution], SpecialCaseSymbolPredicate]
        ] = (),
    ):
        super().__init__(tree_dist)
        self.dsl = dsl
        self.idx_to_name = []
        for x, _ in self.tree_dist.symbols:
            mat = NAME_REGEX.match(x)
            self.idx_to_name.append(mat.group("name") if mat else None)

        self.special_case_predicates = [
            fn(self.tree_dist) for fn in (NameEPredicate, *special_case_predicate_fns)
        ]

        self.handlers = []
        self.config = config

    def currently_defined_indices(self):
        """
        Return the indices of the symbols that are currently defined.
        """
        return self.handlers[-1].currently_defined_indices()

    def compute_mask(self, position: int, symbols: List[int]) -> List[bool]:
        """
        Compute the mask for the given position and symbols. If the last handler is
            defining, then all symbols are valid. Otherwise, only the symbols that
            match the handler's names are valid.
        """
        return self.handlers[-1].compute_mask(
            position, symbols, self.idx_to_name, self.special_case_predicates
        )

    def on_entry(self, position: int, symbol: int) -> Callable[[], None]:
        """
        Updates the stack of handlers when entering a node.
        """
        if not self.handlers:
            assert position == symbol == 0
            self.handlers.append(DefaultHandler(self, [], self.config))
            return self.handlers.pop
        new_handler, undo = self.handlers[-1].on_child_enter(position, symbol)
        self.handlers.append(new_handler)

        return chain_undos(
            [
                getattr(new_handler, "__undo__init__", lambda: None),
                undo,
                self.handlers.pop,
            ]
        )

    def on_exit(self, position: int, symbol: int) -> Callable[[], None]:
        """
        Updates the stack of handlers when exiting a node.
        """
        popped = self.handlers.pop()
        undo_pop = lambda: self.handlers.append(popped)
        if not self.handlers:
            assert position == symbol == 0
            return undo_pop
        undo = self.handlers[-1].on_child_exit(position, symbol, popped)

        return chain_undos([undo_pop, undo])

    def with_handler(self, handler_fn):
        """
        With a new handler function, return a new mask.
        """
        mask_copy = copy.copy(self)
        handler = handler_fn(mask_copy)
        mask_copy.handlers = [handler]
        return mask_copy

    def id_to_name_and_arity(self, symbol_id):
        """
        Convert the symbol ID to a string of the symbol name, and the arity of the symbol.
        """

        if isinstance(symbol_id, ExtraVar):
            return symbol_id.leaf_name(), 0
        return self.tree_dist.symbols[symbol_id]

    def id_to_name(self, symbol_id):
        """
        Convert the symbol ID to a string.
        """
        return self.id_to_name_and_arity(symbol_id)[0]

    def name_to_id(self, name: str):
        """
        Convert the string to a symbol ID.
        """
        if canonicalized_python_name_leaf_regex.match(name):
            assert name not in self.tree_dist.symbol_to_index
            evar = ExtraVar.from_name(name)
            if evar is not None:
                return evar
        return self.tree_dist.symbol_to_index[name]

    @property
    def can_cache(self) -> bool:
        return False

    def cache_key(self, parents: Tuple[Tuple[int, int], ...]) -> Any:
        raise NotImplementedError
