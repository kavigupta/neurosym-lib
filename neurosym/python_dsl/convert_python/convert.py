import ast
from typing import Callable, Dict, List, Union

from frozendict import frozendict

from neurosym.programs.s_expression import SExpression
from neurosym.programs.s_expression_render import render_s_expression
from neurosym.python_dsl.convert_python.parse_python import python_to_python_ast
from neurosym.python_dsl.convert_python.parse_s_exp import s_exp_to_python_ast
from neurosym.python_dsl.convert_python.python_ast import PythonAST


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
