import ast
import base64
from typing import Callable, Dict, List, Union

from frozendict import frozendict

from neurosym.programs.s_expression import SExpression
from neurosym.programs.s_expression_render import parse_s_expression
from neurosym.python_dsl import python_ast_tools
from neurosym.python_dsl.names import PYTHON_DSL_SEPARATOR
from neurosym.utils.documentation import internal_only

from .python_ast import (
    LeafAST,
    ListAST,
    NodeAST,
    PythonAST,
    SequenceAST,
    SliceElementAST,
    SpliceAST,
    StarrableElementAST,
)
from .symbol import PythonSymbol


@internal_only
def s_exp_leaf_to_value(x):
    """
    Returns (True, a python representation of the leaf) if it is a leaf,
        or (False, None) otherwise.
    """
    sym_x = PythonSymbol.parse(x)
    if sym_x is not None:
        return True, sym_x
    if x == "Ellipsis":
        return True, Ellipsis
    if x in {"True", "False", "None"}:
        return True, ast.literal_eval(x)
    if x.startswith("i"):
        return True, int(x[1:])
    if x.startswith("f"):
        return True, float(x[1:])
    if x.startswith("j"):
        return True, complex(x[1:])
    if x.startswith("s_"):
        return True, x[2:]
    if x.startswith("s-"):
        return True, "".join(
            chr(x)
            for x in ast.literal_eval(
                base64.b64decode(x[2:].encode("utf-8")).decode("ascii")
            )
        )
    if x.startswith("b"):
        return True, base64.b64decode(x[1:].encode("utf-8"))

    return False, None


@internal_only
def handle_leaf(
    x: str,
    node_hooks: Dict[str, Callable[[str, List[PythonAST]], PythonAST]],
):
    for hook_prefix, hook in node_hooks.items():
        if x.startswith(hook_prefix):
            return hook(x, [])

    is_leaf, leaf = s_exp_leaf_to_value(x)
    if is_leaf:
        return LeafAST(leaf)
    typ = getattr(ast, x)
    assert not python_ast_tools.fields_for_node(typ), typ
    return NodeAST(typ, [])


@internal_only
def s_exp_to_parsed_ast(
    x: SExpression,
    node_hooks: Dict[str, Callable[[str, List[PythonAST]], PythonAST]],
) -> PythonAST:
    """
    Convert an s-expression (as pairs) to a parsed AST
    """
    if x == "nil":
        return ListAST([])
    if isinstance(x, str):
        return handle_leaf(x, node_hooks)
    assert isinstance(x, SExpression), str((type(x), x))
    tag, args = x.symbol, x.children
    # remove any type information
    tag = tag.split(PYTHON_DSL_SEPARATOR)[0]
    if tag.startswith("const-"):
        assert len(args) == 0
        is_leaf, leaf = s_exp_leaf_to_value(tag[len("const-") :])
        assert is_leaf
        return LeafAST(leaf)
    assert isinstance(tag, str), str(tag)
    args = [s_exp_to_parsed_ast(x, node_hooks) for x in args]
    if tag in {"/seq", "/subseq", "/choiceseq"}:
        return SequenceAST(tag, args)
    if tag in {"/splice"}:
        [arg] = args
        return SpliceAST(arg)
    if tag.startswith("_slice"):
        assert len(args) == 1
        return SliceElementAST(args[0])
    if tag.startswith("_starred"):
        assert len(args) == 1
        return StarrableElementAST(args[0])
    if tag in {"list"}:
        return ListAST(args)
    for hook_prefix, hook in node_hooks.items():
        if tag.startswith(hook_prefix):
            return hook(tag, args)
    return NodeAST(getattr(ast, tag), args)


def s_exp_to_python_ast(
    code: Union[str, SExpression],
    node_hooks: Dict[str, Callable[[str, List[PythonAST]], PythonAST]] = frozendict(),
) -> PythonAST:
    """
    Converts an s expression to a PythonAST object. If the code is a string,
    it is first parsed into an SExpression.

    :param code: The code to convert.
    :param node_hooks: A dictionary of node hooks to use.
    """
    if isinstance(code, str):
        code = parse_s_expression(code)
    code = s_exp_to_parsed_ast(code, node_hooks)
    return code
