import ast
import uuid
from dataclasses import dataclass
from typing import Union

import ast_scope
from no_toplevel_code import wrap_ast

from neurosym.python_dsl import python_ast_tools
from neurosym.utils.documentation import internal_only


@dataclass(frozen=True)
class PythonSymbol:
    """
    Represents a symbol, like &x:3. This means the symbol x in static frame 3.
    Can also represent a global symbol that's either a builtin or an imported
    value. This differs from a symbol defined in the block of code that happens
    to be in global scope, which will be given a static frame number.

    :param name: The name of the symbol.
    :param scope: The scope of the symbol, or None if it's a global symbol.
    """

    name: str
    scope: Union[int, str, None]

    def __post_init__(self):
        scope = self.scope
        if scope is None:
            return
        if isinstance(scope, str):
            assert scope.isdigit(), f"scope must be int or None, got {scope!r}"
        else:
            assert isinstance(scope, int), f"scope must be int or None, got {scope!r}"

    @classmethod
    def parse(cls, x):
        """
        Parses a symbol.
        """
        if x.startswith("&"):
            name, scope = x[1:].split(":")
            return cls(name, scope)
        if x.startswith("g"):
            assert x.startswith("g_")
            return cls(x[2:], None)
        return None

    def render_symbol(self):
        """
        Render this symbol with scope information.
        """
        if self.scope is None:
            return f"g_{self.name}"
        return f"&{self.name}:{self.scope}"


@internal_only
def create_descoper(code):
    """
    Creates a mapping from nodes to numerical ids for scopes.

    Args:
        code: The code.

    Returns:
        The descoper.
    """
    globs = _true_globals(code)
    annot = ast_scope.annotate(code)
    scopes = []
    results = {}
    for node in ast.walk(code):
        if node in annot:
            if node in globs:
                results[node] = None
                continue
            if annot[node] not in scopes:
                scopes.append(annot[node])
            results[node] = scopes.index(annot[node])
    return results


def _true_globals(node):
    """
    Get the true globals of a program.

    Args:
        node: The node.

    Returns:
        The true globals.
    """
    name = "_" + uuid.uuid4().hex
    wpd = wrap_ast(node, name)
    scope_info = ast_scope.annotate(wpd)
    return {
        x
        for x in scope_info
        if scope_info[x] == scope_info.global_scope
        if getattr(x, python_ast_tools.name_field(x)) != name
    }
