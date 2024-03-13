from typing import List

from neurosym.types.type_with_environment import Environment, TypeWithEnvironment

from .preorder_mask import PreorderMask


class TypePreorderMask(PreorderMask):
    """
    Masks out productions that would lead to ill-typed programs.
    """

    def __init__(self, tree_dist, dsl):
        super().__init__(tree_dist)
        self.dsl = dsl
        assert len(self.dsl.valid_root_types) == 1, "Only one root type is supported"
        self.root_type = self.dsl.valid_root_types[0]

        # stack of list of type_with_environment objects
        # each list represents the types of the children of the current node
        self.type_stack: List[List[TypeWithEnvironment]] = []

    def compute_mask(self, position, symbols):
        valid_productions = {
            self.tree_dist.symbol_to_index[sym.symbol()]
            for sym, _ in self.dsl.productions_for_type(self.type_stack[-1][position])
        }
        return [i in valid_productions for i in symbols]

    def on_entry(self, position, symbol):
        symbol, arity = self.tree_dist.symbols[symbol]
        if symbol == "<root>":
            self.type_stack.append(
                [TypeWithEnvironment(self.root_type, Environment.empty())]
            )
            return
        parent_type = self.type_stack[-1][position]
        production = self.dsl.get_production(symbol)
        children_types = production.type_signature().unify_return(parent_type)
        if children_types is None:
            raise ValueError(
                f"Type mismatch in production {production} with parent type {parent_type}"
            )
        assert len(children_types) == arity
        self.type_stack.append(children_types)

    def on_exit(self, position, symbol):
        del position
        del symbol
        self.type_stack.pop()
