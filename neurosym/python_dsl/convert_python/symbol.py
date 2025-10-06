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
        valid = (
            isinstance(scope, str)
            and scope.isdigit()
            or isinstance(scope, (int, _nonsymbol_scope_id))
        )
        assert valid, (
            f"Invalid scope {scope} for symbol {self.name}. "
            "Scope should be an int, str that is a digit, or import_scope_id."
        )

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
            x = x[2:]
            if ":" in x:
                name, scope = x.split(":")
                return cls(name, _nonsymbol_scope_id(int(scope)))
            return cls(x, _nonsymbol_scope_id(None))
        return None

    def render_symbol(self):
        """
        Render this symbol with scope information.
        """
        if isinstance(self.scope, _nonsymbol_scope_id):
            return f"g_{self.name}{self.scope.render()}"
        return f"&{self.name}:{self.scope}"


@dataclass(frozen=True)
class _nonsymbol_scope_id:
    scope: Union[int, None]

    def __post_init__(self):
        assert self.scope is None or isinstance(self.scope, int)

    @classmethod
    @internal_only
    def wrap(cls, scope):
        if isinstance(scope, cls):
            return scope
        return cls(scope)

    @internal_only
    def render(self):
        """
        Render this scope id.
        """
        if self.scope is None:
            return ""
        return f":{self.scope}"


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
                results[node] = _nonsymbol_scope_id(None)
                continue
            if annot[node] not in scopes:
                scopes.append(annot[node])
            results[node] = scopes.index(annot[node])
    # anything that's imported should not be a symbol. This isn't great
    # because it means that import os as x doesn't have x be the symbol,
    # but it is a good first approximation.
    node_to_id_scope = {
        node: (idx, getattr(node, python_ast_tools.name_field(node)))
        for node, idx in results.items()
    }
    import_node_ids = {
        idx_scope
        for node, idx_scope in node_to_id_scope.items()
        if isinstance(node, ast.alias)
    }
    # Make the changes to all nodes that are the same as one that is imported
    for node, idx_scope in node_to_id_scope.items():
        if idx_scope in import_node_ids:
            results[node] = _nonsymbol_scope_id.wrap(results[node])
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
