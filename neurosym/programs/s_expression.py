from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True, eq=True)
class SExpression:
    symbol: str
    children: Tuple["SExpression"]

    @property
    def postorder(self):
        for x in self.children:
            yield from x.postorder
        yield self

    def replace_symbols_by_id(self, id_to_symbol):
        """
        Replace symbols in the S-expression based on the given mapping.

        Args:
            id_to_symbol: A mapping from id(node) to a new symbol.
        """
        return SExpression(
            id_to_symbol.get(id(self), self.symbol),
            tuple(child.replace_symbols_by_id(id_to_symbol) for child in self.children),
        )

    def replace_nodes_by_id(self, id_to_new_node):
        """
        Replace nodes in the S-expression based on the given mapping. Any remapped
            node will not have its children replaced.

        Args:
            id_to_new_node: A mapping from id(node) to a new node.
        """
        if id(self) in id_to_new_node:
            return id_to_new_node[id(self)]
        return SExpression(
            self.symbol,
            tuple(child.replace_nodes_by_id(id_to_new_node) for child in self.children),
        )


@dataclass
class InitializedSExpression:
    symbol: str
    children: Tuple["InitializedSExpression"]
    # state includes things related to the execution of the program,
    # e.g. weights of a neural network
    state: Dict[str, object]

    def all_state_values(self):
        yield from self.state.values()
        for child in self.children:
            yield from child.all_state_values()
