import ast
import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from frozendict import frozendict

from neurosym.programs.s_expression import SExpression

from .splice import Splice
from .symbol import PythonSymbol


class PythonAST(ABC):
    """
    Represents a Python AST. This is a tree-like structure that can be converted to and from
    Python ASTs and s-expressions.
    """

    @abstractmethod
    def to_ns_s_exp(self, config=frozendict()) -> SExpression:
        """
        Convert this PythonAST into an SExpression object

        :param config: A configuration dictionary. This can contain the key
            ``no_leaves``: If True, then leaf nodes will be replaced with a placeholder.
        """

    def to_python(self) -> str:
        """
        Convert this PythonAST into python code.
        """
        code = self.to_python_ast()
        if isinstance(code, Splice):
            code = code.target
        return ast.unparse(code)

    @abstractmethod
    def to_python_ast(self) -> ast.AST:
        """
        Convert this PythonAST into an ast.AST object.
        """

    @abstractmethod
    def map(self, fn):
        """
        Map the given function over this PythonAST. fn is run in post-order,
        i.e., run on all the children and then on the new object.
        """


@dataclass
class NodeAST(PythonAST):
    """
    The NodeAST represents a node in the Python AST. It has a type and a list of children.

    :param typ: The type of the node, which should be a subclass of ast.AST.
    :param children: The children of the node.
    """

    typ: type
    children: List[PythonAST]

    def to_ns_s_exp(self, config=frozendict()):
        if not self.children and not config.get("no_leaves", False):
            return self.typ.__name__

        return SExpression(
            self.typ.__name__, [x.to_ns_s_exp(config) for x in self.children]
        )

    def to_python_ast(self):
        out = self.typ(*[x.to_python_ast() for x in self.children])
        out.lineno = 0
        return out

    def map(self, fn):
        return fn(NodeAST(self.typ, [x.map(fn) for x in self.children]))


@dataclass
class SequenceAST(PythonAST):
    """
    Represents a sequence within the PythonAST, to represent a sequence of statements
    in a body.

    :param head: The head of the sequence (either /seq or /subseq)
    :param elements: The elements of the sequence.
    """

    head: str
    elements: List[PythonAST]

    def __post_init__(self):
        assert isinstance(self.head, str), self.head
        assert all(isinstance(x, PythonAST) for x in self.elements), self.elements

    def to_ns_s_exp(self, config=frozendict()):
        return SExpression(self.head, [x.to_ns_s_exp(config) for x in self.elements])

    def to_python_ast(self):
        result = []
        for x in self.elements:
            x = x.to_python_ast()
            if isinstance(x, Splice):
                result += x.target
            else:
                result += [x]
        return result

    def map(self, fn):
        return fn(SequenceAST(self.head, [x.map(fn) for x in self.elements]))


@dataclass
class ListAST(PythonAST):
    """
    Represents a list in the Python AST that is not a sequence.

    :param children: The children of the list.
    """

    children: List[PythonAST]

    def to_ns_s_exp(self, config=frozendict()):
        if not self.children:
            return SExpression("list", []) if config.get("no_leaves", False) else "nil"

        return SExpression("list", [x.to_ns_s_exp(config) for x in self.children])

    def to_python_ast(self):
        return [x.to_python_ast() for x in self.children]

    def map(self, fn):
        return fn(ListAST([x.map(fn) for x in self.children]))


@dataclass
class LeafAST(PythonAST):
    """
    Represents a leaf in the Python AST. This is a leaf node that is not an ast.AST object.

    :param leaf: The leaf object.
    """

    leaf: object

    def __post_init__(self):
        assert not isinstance(self.leaf, PythonAST)

    def to_ns_s_exp(self, config=frozendict()):
        leaf_as_string = self._render_leaf_as_string()
        if not config.get("no_leaves", False):
            return leaf_as_string
        return SExpression("const-" + leaf_as_string, [])

    def _render_leaf_as_string(self):
        if (
            self.leaf is True
            or self.leaf is False
            or self.leaf is None
            or self.leaf is Ellipsis
        ):
            return str(self.leaf)
        if isinstance(self.leaf, PythonSymbol):
            return self.leaf.render_symbol()
        if isinstance(self.leaf, float):
            return f"f{self.leaf}"
        if isinstance(self.leaf, int):
            return f"i{self.leaf}"
        if isinstance(self.leaf, complex):
            return f"j{self.leaf}"
        if isinstance(self.leaf, str):
            # if all are renderable directly without whitespace, just use that
            if all(
                c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_."
                for c in self.leaf
            ):
                return "s_" + self.leaf
            return "s-" + base64.b64encode(
                str([ord(x) for x in self.leaf]).encode("ascii")
            ).decode("utf-8")
        if isinstance(self.leaf, bytes):
            return "b" + base64.b64encode(self.leaf).decode("utf-8")
        raise RuntimeError(f"invalid leaf: {self.leaf}")

    def to_python_ast(self):
        if isinstance(self.leaf, PythonSymbol):
            return self.leaf.name
        return self.leaf

    def map(self, fn):
        return fn(LeafAST(self.leaf))


@dataclass
class SliceElementAST(PythonAST):
    """
    Represents a slice element, which is an expression inside a slice.
        This needs to be a separate class because some expressions are
        only valid inside slices.

    :param content: The content of the slice element.
    """

    content: PythonAST

    @classmethod
    def of(cls, x):
        if isinstance(x, SliceElementAST):
            return x
        return SliceElementAST(x)

    def to_ns_s_exp(self, config=frozendict()):
        # should not be necessary; since we have the assertion
        # but pylint is not smart enough to figure that out
        # pylint: disable=no-member
        content = self.content
        if isinstance(self.content, StarrableElementAST):
            # safe because it is not actually legal to have a starred element
            # in a slice
            content = content.content
        if isinstance(content, NodeAST):
            if content.typ is ast.Slice:
                return SExpression("_slice_slice", [content.to_ns_s_exp(config)])
            if content.typ is ast.Tuple:
                assert isinstance(content.children, list)
                assert len(content.children) == 2
                content_children = list(content.children)
                content_children[0] = ListAST(
                    [SliceElementAST.of(x) for x in content_children[0].children]
                )
                content = NodeAST(typ=ast.Tuple, children=content_children)

                return SExpression("_slice_tuple", [content.to_ns_s_exp(config)])
        return SExpression("_slice_content", [content.to_ns_s_exp(config)])

    def to_python_ast(self):
        return self.content.to_python_ast()

    def map(self, fn):
        return fn(SliceElementAST(self.content.map(fn)))


@dataclass
class StarrableElementAST(PythonAST):
    """
    Represents a starrable element, which is an expression that can be starred.

    :param content: The content of the starrable element.
    """

    content: PythonAST

    def to_ns_s_exp(self, config=frozendict()):
        # pylint: disable=no-member
        if isinstance(self.content, NodeAST) and self.content.typ is ast.Starred:
            return SExpression("_starred_starred", [self.content.to_ns_s_exp(config)])
        return SExpression("_starred_content", [self.content.to_ns_s_exp(config)])

    def to_python_ast(self):
        return self.content.to_python_ast()

    def map(self, fn):
        return fn(StarrableElementAST(self.content.map(fn)))


@dataclass
class SpliceAST(PythonAST):
    """
    Represents a splice in the Python AST. This is a special node that is used to splice
    the children of the node into the parent, which must be a sequence.

    :param content: The content of the splice. This is either a sequence itself, or a node
        representing something that is a sequence but not explicitly represented as such (e.g.,
        an abstraction that returns a sequence).

    :param content: The content to splice.
    """

    content: PythonAST

    def to_ns_s_exp(self, config=frozendict()):
        return SExpression("/splice", [self.content.to_ns_s_exp(config)])

    def to_python_ast(self):
        return Splice(self.content.to_python_ast())

    def map(self, fn):
        return fn(SpliceAST(self.content.map(fn)))
