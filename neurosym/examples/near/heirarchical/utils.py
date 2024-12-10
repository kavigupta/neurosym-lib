from typing import Tuple

from neurosym.programs.s_expression import InitializedSExpression


def replace_first(
    ip: InitializedSExpression, symbol: str, replacement: InitializedSExpression
) -> Tuple[InitializedSExpression, bool]:
    """
    Replace the first occurrence of a node with a given symbol in an InitializedSExpression
    with a replacement.

    :param ip: The InitializedSExpression to replace the symbol in.
    :param symbol: The symbol whose node's first occurrence to replace.
    :param replacement: The value to replace the node with.

    :return: A tuple of the new InitializedSExpression and a boolean indicating whether
    the replacement was successful. The replacement is successful if the symbol was found
    in the tree.
    """
    if ip.symbol == symbol:
        return replacement, True
    new_children = []
    replaced = False
    for child in ip.children:
        if replaced:
            new_children.append(child)
            continue
        new_child, child_replaced = replace_first(child, symbol, replacement)
        replaced = replaced or child_replaced
        new_children.append(new_child)
    return InitializedSExpression(ip.symbol, new_children, ip.state), replaced
