import ast

from neurosym.python_dsl.names import PYTHON_DSL_SEPARATOR


def fields_for_node(node):
    """
    Get the fields for a node. If the node is a string, get the
        segment before the python dsl separator
    """
    if isinstance(node, str):
        node = node.split(PYTHON_DSL_SEPARATOR)[0]
        node = getattr(ast, node)

    return node._fields
