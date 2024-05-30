import ast

from neurosym.python_dsl.names import PYTHON_DSL_SEPARATOR
from neurosym.types.type import AtomicType, ListType
from neurosym.types.type_string_repr import parse_type

# types we do not care about, so just send them to the same state
pruned_python_dfa_states = ["TA"]


def fields_for_node(node):
    """
    Get the fields for a node. If the node is a string, get the
        segment before the python dsl separator
    """
    if isinstance(node, str):
        node = node.split(PYTHON_DSL_SEPARATOR)[0]
        node = getattr(ast, node)

    return node._fields


def field_is_body(node_type: type, field_name: str):
    """
    Returns whether a field is a body field.

    Args:
        node_type: The type of the node.
        field_name: The name of the field.

    Returns:
        Whether the field is a body field.
    """
    assert isinstance(node_type, type)
    if node_type == ast.IfExp:
        return False  # not body of statements
    if node_type == ast.Lambda:
        return False  # not body of statements
    return field_name in {"body", "orelse", "finalbody"}


def field_is_starrable(node_type, field_name):
    """
    Field is starrable if it is a list of elements or a call with args.

    Args:
        node_type: The type of the node.
        field_name: The name of the field.

    Returns:
        Whether the field is starrable.
    """
    if field_name == "elts":
        assert node_type in {
            ast.List,
            ast.Tuple,
            ast.Set,
        }, (node_type, field_name)
        return True
    return node_type == ast.Call and field_name == "args"


def name_field(node: ast.AST):
    """
    Find the name field for a node.
    """
    t = type(node)
    if t == ast.Name:
        return "id"
    if t == ast.arg:
        return "arg"
    if t == ast.FunctionDef:
        return "name"
    if t == ast.AsyncFunctionDef:
        return "name"
    if t == ast.ExceptHandler:
        return "name" if node.name is not None else None
    if t == ast.ClassDef:
        return "name"
    if t == ast.alias:
        if node.asname is None:
            return "name"
        return "asname"
    raise NotImplementedError(f"Unexpected type: {t}")


def is_sequence_type(x):
    """
    Returns whether a type is a sequence type.
    Sequence types are either lists or seqS.
    """
    x = parse_type(x)
    if isinstance(x, ListType):
        return True
    if not isinstance(x, AtomicType):
        return False
    return x.name == "seqS"


def is_sequence_symbol(x):
    """
    Returns whether a symbol is a sequence symbol.
    """
    return x in ["/seq", "/subseq", "list", "/choiceseq"]


def is_sequence(type_name, head_symbol):
    """
    Returns whether a given type and head symbol correspond to a sequence.
        If there is a mismatch between the type and the head symbol, this function
        errors.

    Args:
        type_name: The type name.
        head_symbol: The head symbol.
    """

    seq_type = is_sequence_type(type_name)
    seq_symbol = is_sequence_symbol(head_symbol)
    if type_name in pruned_python_dfa_states:
        return seq_symbol
    return seq_type and seq_symbol


def clean_type(x):
    """
    Replace [] with __ in the type name
    """
    return x.replace("[", "_").replace("]", "_")


def unclean_type(x):
    """
    Replace __ with [] in the type name
    """
    if "_" not in x:
        return x
    assert x.count("_") == 2, x
    return x.replace("_", "[", 1).replace("_", "]", 1)
