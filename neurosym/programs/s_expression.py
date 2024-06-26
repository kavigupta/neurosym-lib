from dataclasses import dataclass
from typing import Dict, Tuple, Union


@dataclass(frozen=True, eq=True)
class SExpression:
    """
    Represents an s-expression, that is, an expression that contains a
    symbol and a list of children, which are also s-expressions or strings
    (leaves).

    :field symbol: The symbol of the s-expression.
    :field children: A tuple of children of the s-expression. Each child can
        be either a string or another s-expression.
    """

    symbol: str
    children: Tuple[Union["SExpression", str]]

    @property
    def postorder(self):
        for x in self.children:
            yield from x.postorder
        yield self

    def replace_symbols_by_id(self, id_to_symbol):
        """
        Replace symbols in the S-expression based on the given mapping. The current
        SExpression will not be mutated

        :param id_to_symbol: A mapping from id(node) to a new symbol.

        :return: A new S-expression with the symbols replaced.
        """
        return SExpression(
            id_to_symbol.get(id(self), self.symbol),
            tuple(child.replace_symbols_by_id(id_to_symbol) for child in self.children),
        )

    def replace_nodes_by_id(self, id_to_new_node):
        """
        Replace nodes in the S-expression based on the given mapping. Any remapped
        node will not have its children replaced. The current SExpression will
        not be mutated

        :param id_to_new_node: A mapping from id(node) to a new node.
        :return: A new S-expression with the nodes replaced.
        """
        if id(self) in id_to_new_node:
            return id_to_new_node[id(self)]
        return SExpression(
            self.symbol,
            tuple(child.replace_nodes_by_id(id_to_new_node) for child in self.children),
        )


@dataclass
class InitializedSExpression:
    """
    Represents a SExpression with an additional state. This is useful for
    representing a program with a state, e.g. a neural network with weights.

    :field symbol: The symbol of the s-expression.
    :field children: A tuple of children of the s-expression. Each child can
        be either a string or another s-expression.
    :field state: A dictionary of state values.
    """

    symbol: str
    children: Tuple["InitializedSExpression"]
    # state includes things related to the execution of the program,
    # e.g. weights of a neural network
    state: Dict[str, object]

    def all_state_values(self):
        """
        Yield all state values in this InitializedSExpression and its children.

        :return: An iterator over all state values.
        """
        yield from self.state.values()
        for child in self.children:
            yield from child.all_state_values()
