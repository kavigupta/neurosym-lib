import ast
from types import NoneType
from typing import List, Union

from neurosym.python_dsl import python_ast_tools

from .python_ast import (
    LeafAST,
    ListAST,
    NodeAST,
    PythonAST,
    SequenceAST,
    SliceElementAST,
    StarrableElementAST,
)
from .symbol import PythonSymbol, create_descoper


def _python_body_to_parsed_ast(x: List[ast.AST], descoper: dict) -> SequenceAST:
    """
    Convert a python body to a PythonAST object.

    Args:
        x: The python body.
        descoper: The descoper to use.

    Returns:
        The parsed PythonAST.
    """
    assert isinstance(x, list), str(x)
    x = [_python_ast_to_parsed_ast(x, descoper) for x in x]
    return SequenceAST("/seq", x)


def _python_ast_to_parsed_ast(x, descoper: dict) -> PythonAST:
    """
    Convert an ast.AST object to a PythonAST object.

    Args:
        x: The ast.AST object.
        descoper: The descoper to use.

    Returns:
        The parsed PythonAST.
    """
    if isinstance(x, ast.AST):
        result = []
        for f in python_ast_tools.fields_for_node(x):
            el = getattr(x, f)
            if x in descoper and f == python_ast_tools.name_field(x):
                assert isinstance(el, str), (x, f, el)
                result.append(LeafAST(PythonSymbol(el, descoper[x])))
            else:
                if f == "slice":
                    result.append(
                        SliceElementAST(_python_ast_to_parsed_ast(el, descoper))
                    )
                elif python_ast_tools.field_is_starrable(type(x), f):
                    out = _python_ast_to_parsed_ast(el, descoper)
                    out = ListAST([StarrableElementAST(x) for x in out.children])
                    result.append(out)
                elif python_ast_tools.field_is_body(type(x), f):
                    result.append(_python_body_to_parsed_ast(el, descoper))
                else:
                    result.append(_python_ast_to_parsed_ast(el, descoper))
        return NodeAST(type(x), result)
    if isinstance(x, list):
        return ListAST([_python_ast_to_parsed_ast(x, descoper) for x in x])
    if x is None or x is Ellipsis or isinstance(x, (int, float, complex, str, bytes)):
        return LeafAST(x)
    raise ValueError(f"Unsupported node {x}")


def python_to_python_ast(
    code: Union[str, ast.AST], descoper: Union[NoneType, dict] = None
) -> PythonAST:
    """
    Parse the given python code into a PythonAST. If the code is a string,
        it is first parsed into an AST.

    :param code: The python code to parse.
    :param descoper: The descoper to use. If None, a new one is created.

    :return: The parsed PythonAST.
    """

    if isinstance(code, str):
        code = ast.parse(code)
    code = _python_ast_to_parsed_ast(
        code,
        descoper if descoper is not None else create_descoper(code),
    )
    return code
