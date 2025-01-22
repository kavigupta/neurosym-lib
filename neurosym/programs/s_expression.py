import uuid
from dataclasses import dataclass, field, replace
from typing import Dict, Tuple, Union

from neurosym.utils.documentation import internal_only


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

    def replace_first(
        self, symbol: str, replacement: "SExpression"
    ) -> Tuple["SExpression", bool]:
        """
        Replace the first occurrence of a node with a given symbol in this SExpression
        with a replacement.

        In general, a minimal number of nodes should be copied. If the symbol is not found,
        a reference to this SExpression is returned.

        :param symbol: The symbol whose node's first occurrence to replace.
        :param replacement: The value to replace the node with.

        :return: A tuple of the new SExpression and a boolean indicating whether
            the replacement was successful. The replacement is successful if the symbol was found
            in the tree.
        """
        return _replace_first(self, symbol, replacement)


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
    ident: uuid.UUID = field(default_factory=uuid.uuid4)

    def all_state_values(self):
        """
        Yield all state values in this InitializedSExpression and its children.

        :return: An iterator over all state values.
        """
        yield from self.state.values()
        for child in self.children:
            yield from child.all_state_values()

    def uninitialize(self) -> SExpression:
        """
        Return the SExpression corresponding to this InitializedSExpression.

        :return: The SExpression corresponding to this InitializedSExpression.
        """

        return SExpression(
            self.symbol,
            tuple(child.uninitialize() for child in self.children),
        )

    def replace_first(
        self, symbol: str, replacement: "InitializedSExpression"
    ) -> Tuple["InitializedSExpression", bool]:
        """
        Replace the first occurrence of a node with a given symbol in this InitializedSExpression
        with a replacement.

        In general, a minimal number of nodes should be copied. If the symbol is not found,
        a reference to this InitializedSExpression is returned.

        :param symbol: The symbol whose node's first occurrence to replace.
        :param replacement: The value to replace the node with.

        :return: A tuple of the new InitializedSExpression and a boolean indicating whether
            the replacement was successful. The replacement is successful if the symbol was found
            in the tree.
        """
        return _replace_first(self, symbol, replacement)

    def __hash__(self):
        return hash(self.ident)


def _replace_first(
    s_exp, symbol: str, replacement: InitializedSExpression | SExpression
) -> Tuple[InitializedSExpression | SExpression, bool]:
    if s_exp.symbol == symbol:
        return replacement, True
    new_children = []
    replaced = False
    for child in s_exp.children:
        if replaced:
            new_children.append(child)
            continue
        new_child, child_replaced = _replace_first(child, symbol, replacement)
        replaced = replaced or child_replaced
        new_children.append(new_child)
    if not replaced:
        return s_exp, False
    return (
        # type(self)(self.symbol, tuple(new_children), self.state),
        replace(s_exp, children=tuple(new_children)),
        replaced,
    )


@internal_only
def is_initialized_s_expression(p):
    """
    Check if a value is an InitializedSExpression. Duck typed because
    we want to allow Holes and other classes to be treated as InitializedSExpressions.
    """
    return hasattr(p, "all_state_values")


def postorder(s_exp: SExpression | InitializedSExpression):
    """
    Traverse an SExpression/InitializedSExpression in postorder.

    :param s_exp: The expression to traverse.

    :return: An iterator over the expression in postorder.
    """
    for x in s_exp.children:
        yield from postorder(x)
    yield s_exp
