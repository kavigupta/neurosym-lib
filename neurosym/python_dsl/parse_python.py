import ast

from .python_ast import (
    LeafAST,
    ListAST,
    NodeAST,
    SequenceAST,
    SliceElementAST,
    StarrableElementAST,
)
from .python_ast_tools import (
    field_is_body,
    field_is_starrable,
    fields_for_node,
    name_field,
)
from .symbol import PythonSymbol


def python_body_to_parsed_ast(x, descoper):
    assert isinstance(x, list), str(x)
    x = [python_ast_to_parsed_ast(x, descoper) for x in x]
    return SequenceAST("/seq", x)


def python_ast_to_parsed_ast(x, descoper):
    """
    Convert an ast.AST object to a PythonAST object.
    """
    if isinstance(x, ast.AST):
        result = []
        for f in fields_for_node(x):
            el = getattr(x, f)
            if x in descoper and f == name_field(x):
                assert isinstance(el, str), (x, f, el)
                result.append(LeafAST(PythonSymbol(el, descoper[x])))
            else:
                if f == "slice":
                    result.append(
                        SliceElementAST(python_ast_to_parsed_ast(el, descoper))
                    )
                elif field_is_starrable(type(x), f):
                    out = python_ast_to_parsed_ast(el, descoper)
                    out = ListAST([StarrableElementAST(x) for x in out.children])
                    result.append(out)
                elif field_is_body(type(x), f):
                    result.append(python_body_to_parsed_ast(el, descoper))
                else:
                    result.append(python_ast_to_parsed_ast(el, descoper))
        return NodeAST(type(x), result)
    if isinstance(x, list):
        return ListAST([python_ast_to_parsed_ast(x, descoper) for x in x])
    if x is None or x is Ellipsis or isinstance(x, (int, float, complex, str, bytes)):
        return LeafAST(x)
    raise ValueError(f"Unsupported node {x}")
