import ast
from typing import Callable, Dict, List, Union

from frozendict import frozendict

from neurosym.programs.s_expression import SExpression
from neurosym.programs.s_expression_render import render_s_expression
from neurosym.python_dsl.convert_python.parse_python import python_to_python_ast
from neurosym.python_dsl.convert_python.parse_s_exp import s_exp_to_python_ast
from neurosym.python_dsl.convert_python.python_ast import (
    NodeAST,
    PythonAST,
    SequenceAST,
)
from neurosym.python_dsl.run_dfa import add_disambiguating_type_tags


def python_to_s_exp(code: Union[str, ast.AST], **kwargs) -> str:
    """
    Converts python code to an s-expression.
    """
    return render_s_expression(python_to_python_ast(code).to_ns_s_exp(kwargs))


def s_exp_to_python(
    code: Union[str, SExpression],
    node_hooks: Dict[str, Callable[[str, List[PythonAST]], PythonAST]] = frozendict(),
) -> str:
    """
    Converts an s expression to python code.
    """
    return s_exp_to_python_ast(code, node_hooks).to_python()


def to_type_annotated_ns_s_exp(
    code: PythonAST, dfa: dict, start_state: str
) -> SExpression:
    """
    Like to_ns_s_exp, but adds type annotations.
    """
    return add_disambiguating_type_tags(
        dfa, code.to_ns_s_exp(dict(no_leaves=True)), start_state
    )


def python_statement_to_python_ast(code: Union[str, ast.AST]) -> PythonAST:
    """
    Like python_to_python_ast, but for a single statement.
    """
    code = python_statements_to_python_ast(code)
    assert (
        len(code.elements) == 1
    ), f"expected only one statement; got: [{[x.to_python() for x in code.elements]}]]"
    code = code.elements[0]
    return code


def python_statements_to_python_ast(code: Union[str, ast.AST]) -> SequenceAST:
    """
    Like python_to_python_ast, but for a sequence of statements.
    """
    code = python_to_python_ast(code)
    assert isinstance(code, NodeAST) and code.typ is ast.Module
    assert len(code.children) == 2
    code = code.children[0]
    assert isinstance(code, SequenceAST), code
    return code
