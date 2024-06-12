import ast

from neurosym.python_dsl.convert_python.python_ast import (
    LeafAST,
    ListAST,
    NodeAST,
    PythonAST,
)
from neurosym.python_dsl.convert_python.symbol import PythonSymbol


def make_constant(leaf):
    """
    Create a constant PythonAST from the given leaf value (which must be a python constant).
    """
    assert not isinstance(leaf, PythonAST), leaf
    return NodeAST(typ=ast.Constant, children=[LeafAST(leaf=leaf), LeafAST(leaf=None)])


def make_name(name_node):
    """
    Create a name PythonAST from the given name node containing a symbol.
    """
    assert isinstance(name_node, LeafAST) and isinstance(
        name_node.leaf, PythonSymbol
    ), name_node
    return NodeAST(
        typ=ast.Name,
        children=[
            name_node,
            NodeAST(typ=ast.Load, children=[]),
        ],
    )


def make_call(name_sym, *arguments):
    """
    Create a call PythonAST from the given symbol and arguments.

    In this case, the symbol must be a symbol representing a name.
    """
    assert isinstance(name_sym, PythonSymbol), name_sym
    return NodeAST(
        typ=ast.Call,
        children=[
            make_name(LeafAST(name_sym)),
            ListAST(children=arguments),
            ListAST(children=[]),
        ],
    )


def make_expr_stmt(expr):
    """
    Create an expression statement PythonAST from the given expression.
    """
    return NodeAST(typ=ast.Expr, children=[expr])
