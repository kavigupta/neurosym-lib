from neurosym.program_dist.tree_distribution.ordering import DictionaryNodeOrdering
from neurosym.python_dsl.python_ast_tools import fields_for_node


def _field_order(node, fields):
    """
    Get the order of the given fields in a node.
    """
    node_fields = fields_for_node(node)
    assert set(fields) == set(node_fields)
    return [node_fields.index(f) for f in fields]


def python_ordering_dictionary():
    """
    The ordering dictionary for Python nodes. This is a dictionary that maps node types
    to a list of integers. The integers are the indices of the fields in the node
    that should be visited in order.
    """
    fields = [
        ("ListComp~E", ["generators", "elt"]),
        ("GeneratorExp~E", ["generators", "elt"]),
        ("SetComp~E", ["generators", "elt"]),
        ("DictComp~E", ["generators", "key", "value"]),
    ]

    result = {}
    for node, fields in fields:
        result[node] = _field_order(node, fields)
    return result


class PythonNodeOrdering(DictionaryNodeOrdering):
    """
    Orders the subnodes of a node according to a dictionary.

    :param dist: The ``TreeDistribution`` that the ordering is applied to.
    """

    def __init__(self, dist):
        super().__init__(dist, python_ordering_dictionary(), tolerate_missing=True)
