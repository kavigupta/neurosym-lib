import ast
from dataclasses import dataclass

import ast_scope
from imperative_stitch.utils.ast_utils import true_globals


@dataclass(frozen=True)
class PythonSymbol:
    """
    Represents a symbol, like &x:3. This means the symbol x in static frame 3.
    Can also represent a global symbol that's either a builtin or an imported
        value. This differs from a symbol defined in the block of code that happens
        to be in global scope, which will be given a static frame number.
    """

    name: str
    scope: ast_scope.scope.Scope

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

    def render(self):
        if self.scope is None:
            return f"g_{self.name}"
        return f"&{self.name}:{self.scope}"


def create_descoper(code):
    """
    Creates a mapping from nodes to numerical ids for scopes.
    """
    globs = true_globals(code)
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
