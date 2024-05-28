import ast
import base64
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union

from increase_recursionlimit import increase_recursionlimit

from neurosym.programs.s_expression import SExpression
from neurosym.programs.s_expression_render import (
    parse_s_expression,
    render_s_expression,
)

from .splice import Splice
from .symbol import PythonSymbol, create_descoper


class PythonAST(ABC):
    """
    Represents a Parsed AST.
    """

    @classmethod
    def from_python_ast(cls, ast_node, descoper=None):
        """
        Convert the given python AST into a PythonAST.
        """
        # pylint: disable=R0401
        from .parse_python import python_ast_to_parsed_ast

        with increase_recursionlimit():
            return python_ast_to_parsed_ast(
                ast_node,
                descoper if descoper is not None else create_descoper(ast_node),
            )

    @classmethod
    def parse_python_module(cls, code):
        """
        Parse the given python code into a PythonAST.
        """
        # pylint: disable=R0401
        from .parse_python import python_ast_to_parsed_ast

        with increase_recursionlimit():
            code = ast.parse(code)
            code = python_ast_to_parsed_ast(code, create_descoper(code))
            return code

    @classmethod
    def parse_python_statements(cls, code):
        code = cls.parse_python_module(code)
        assert isinstance(code, NodeAST) and code.typ is ast.Module
        assert len(code.children) == 2
        code = code.children[0]
        return code

    @classmethod
    def parse_python_statement(cls, code):
        code = cls.parse_python_statements(code)
        assert isinstance(code, SequenceAST), code
        assert (
            len(code.elements) == 1
        ), f"expected only one statement; got: [{[x.to_python() for x in code.elements]}]]"
        code = code.elements[0]
        return code

    @classmethod
    def parse_s_expression(cls, code):
        """
        Parse the given s-expression into a PythonAST.
        """
        # pylint: disable=R0401
        with increase_recursionlimit():
            from .parse_s_exp import s_exp_to_parsed_ast

            code = parse_s_expression(code)
            code = s_exp_to_parsed_ast(code)
            return code

    def to_s_exp(self, *, renderer_kwargs=None, no_leaves=False) -> SExpression:
        """
        Convert this PythonAST into an s-expression.
        """
        if renderer_kwargs is None:
            renderer_kwargs = {}
        with increase_recursionlimit():
            return render_s_expression(self.to_ns_s_exp(dict(no_leaves=no_leaves)))

    @abstractmethod
    def to_ns_s_exp(self, config):
        """
        Convert this PythonAST into a pair s-expression.
        """

    def to_type_annotated_ns_s_exp(self, dfa, start_state):
        # pylint: disable=cyclic-import
        from imperative_stitch.utils.export_as_dsl import add_disambiguating_type_tags

        return add_disambiguating_type_tags(
            dfa, self.to_ns_s_exp(dict(no_leaves=True)), start_state
        )

    def to_type_annotated_de_bruijn_ns_s_exp(
        self, dfa, start_state, *, abstrs=(), max_explicit_dbvar_index
    ):
        # pylint: disable=cyclic-import
        from imperative_stitch.utils.def_use_mask.canonicalize_de_bruijn import (
            canonicalize_de_bruijn,
        )

        [result] = canonicalize_de_bruijn(
            [self], [start_state], dfa, abstrs, max_explicit_dbvar_index
        )
        return result

    @classmethod
    def from_type_annotated_de_bruijn_ns_s_exp(cls, s_exp, dfa, abstrs=()):
        # pylint: disable=cyclic-import
        from imperative_stitch.utils.def_use_mask.canonicalize_de_bruijn import (
            uncanonicalize_de_bruijn,
        )

        s_exp = uncanonicalize_de_bruijn(dfa, s_exp, abstrs)

        return cls.parse_s_expression(render_s_expression(s_exp))

    def to_python(self):
        """
        Convert this PythonAST into python code.
        """
        with increase_recursionlimit():
            code = self.to_python_ast()
            if isinstance(code, Splice):
                code = code.target
            return ast.unparse(code)

    @abstractmethod
    def to_python_ast(self):
        """
        Convert this PythonAST into a python AST.
        """

    @abstractmethod
    def map(self, fn):
        """
        Map the given function over this PythonAST. fn is run in post-order,
            i.e., run on all the children and then on the new object.
        """

    def _replace_with_substitute(self, arguments):
        """
        Replace this PythonAST with the corresponding argument from the given arguments.
        """
        del arguments
        # by default, do nothing
        return self

    def substitute(self, arguments):
        """
        Substitute the given arguments into this PythonAST.
        """
        # pylint: disable=protected-access
        return self.map(lambda x: x._replace_with_substitute(arguments))

    def _collect_abstraction_calls(self, result):
        """
        Collect all abstraction calls in this PythonAST. Adds them to the given
            dictionary from handle to abstraction call object.
        """
        del result
        # by default, do nothing
        return self

    def abstraction_calls(self):
        """
        Collect all abstraction calls in this PythonAST. Returns a dictionary
            from handle to abstraction call object.
        """
        result = {}
        # pylint: disable=protected-access
        self.map(lambda x: x._collect_abstraction_calls(result))
        return result

    def _replace_abstraction_calls(self, handle_to_replacement):
        """
        Replace the abstraction call with the given handle with the given replacement.
        """
        del handle_to_replacement
        return self

    def replace_abstraction_calls(self, handle_to_replacement):
        """
        Replace the abstraction call with the given handle with the given replacement.
        """
        # pylint: disable=protected-access
        return self.map(lambda x: x._replace_abstraction_calls(handle_to_replacement))

    def map_abstraction_calls(self, replace_fn):
        """
        Map each abstraction call through the given function.
        """
        handle_to_replacement = self.abstraction_calls()
        handle_to_replacement = {
            handle: replace_fn(call) for handle, call in handle_to_replacement.items()
        }
        return self.replace_abstraction_calls(handle_to_replacement)

    def abstraction_calls_to_stubs(self, abstractions):
        """
        Replace all abstraction calls with stubs. Does so via a double iteration.
            Possibly faster to use a linearization of the set of stubs.
        """
        result = self
        while True:
            abstraction_calls = result.abstraction_calls()
            if not abstraction_calls:
                return result
            replacement = {}
            for handle, node in abstraction_calls.items():
                if (set(node.abstraction_calls()) - {handle}) == set():
                    replacement[handle] = abstractions[node.tag].create_stub(node.args)
            result = result.replace_abstraction_calls(replacement)

    def abstraction_calls_to_bodies(self, abstractions, *, pragmas=False):
        """
        Replace all abstraction calls with their bodies.
        """
        return self.map_abstraction_calls(
            lambda call: abstractions[call.tag].substitute_body(
                call.args, pragmas=pragmas
            )
        )

    def abstraction_calls_to_bodies_recursively(self, abstractions, *, pragmas=False):
        """
        Replace all abstraction calls with their bodies, recursively.
        """
        result = self
        while True:
            result = result.abstraction_calls_to_bodies(abstractions, pragmas=pragmas)
            if not result.abstraction_calls():
                return result

    @classmethod
    def constant(cls, leaf):
        """
        Create a constant PythonAST from the given leaf value (which must be a python constant).
        """
        assert not isinstance(leaf, PythonAST), leaf
        return NodeAST(
            typ=ast.Constant, children=[LeafAST(leaf=leaf), LeafAST(leaf=None)]
        )

    @classmethod
    def name(cls, name_node):
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

    @classmethod
    def call(cls, name_sym, *arguments):
        """
        Create a call PythonAST from the given symbol and arguments.

        In this case, the symbol must be a symbol representing a name.
        """
        assert isinstance(name_sym, PythonSymbol), name_sym
        return NodeAST(
            typ=ast.Call,
            children=[
                cls.name(LeafAST(name_sym)),
                ListAST(children=arguments),
                ListAST(children=[]),
            ],
        )

    @classmethod
    def expr_stmt(cls, expr):
        """
        Create an expression statement PythonAST from the given expression.
        """
        return NodeAST(typ=ast.Expr, children=[expr])

    def render_symvar(self):
        """
        Render this PythonAST as a __ref__ variable for stub display, i.e.,
            `a` -> `__ref__(a)`
        """
        return PythonAST.call(
            PythonSymbol(name="__ref__", scope=None), PythonAST.name(self)
        )

    def render_codevar(self):
        """
        Render this PythonAST as a __code__ variable for stub display, i.e.,
            `a` -> `__code__("a")`
        """
        return PythonAST.call(
            PythonSymbol(name="__code__", scope=None),
            PythonAST.constant(self.to_python()),
        )

    def wrap_in_metavariable(self, name):
        return NodeAST(
            ast.Set,
            [
                ListAST(
                    [
                        PythonAST.name(LeafAST(PythonSymbol("__metavariable__", None))),
                        PythonAST.name(LeafAST(PythonSymbol(name, None))),
                        self,
                    ]
                )
            ],
        )

    def wrap_in_choicevar(self):
        return SequenceAST(
            "/seq",
            [
                PythonAST.parse_python_statement("__start_choice__"),
                self,
                PythonAST.parse_python_statement("__end_choice__"),
            ],
        )


@dataclass
class SequenceAST(PythonAST):
    head: str
    elements: List[PythonAST]

    def __post_init__(self):
        assert isinstance(self.head, str), self.head
        assert all(isinstance(x, PythonAST) for x in self.elements), self.elements

    def to_ns_s_exp(self, config):
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
class NodeAST(PythonAST):
    typ: type
    children: List[PythonAST]

    def to_ns_s_exp(self, config):
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
class ListAST(PythonAST):
    children: List[PythonAST]

    def to_ns_s_exp(self, config):
        if not self.children:
            return SExpression("list", []) if config.get("no_leaves", False) else "nil"

        return SExpression("list", [x.to_ns_s_exp(config) for x in self.children])

    def to_python_ast(self):
        return [x.to_python_ast() for x in self.children]

    def map(self, fn):
        return fn(ListAST([x.map(fn) for x in self.children]))


@dataclass
class LeafAST(PythonAST):
    leaf: object

    def __post_init__(self):
        assert not isinstance(self.leaf, PythonAST)

    def to_ns_s_exp(self, config):
        leaf_as_string = self.render_leaf_as_string()
        if not config.get("no_leaves", False):
            return leaf_as_string
        return SExpression("const-" + leaf_as_string, [])

    def render_leaf_as_string(self):
        if (
            self.leaf is True
            or self.leaf is False
            or self.leaf is None
            or self.leaf is Ellipsis
        ):
            return str(self.leaf)
        if isinstance(self.leaf, PythonSymbol):
            return self.leaf.render()
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
class Variable(PythonAST):
    sym: str

    @property
    def idx(self):
        return int(self.sym[1:])

    def map(self, fn):
        return fn(self)

    def to_ns_s_exp(self, config):
        if config.get("no_leaves", False):
            return SExpression("var-" + self.sym, [])
        return self.sym


@dataclass
class SymvarAST(Variable):
    def to_python_ast(self):
        return self.sym

    def _replace_with_substitute(self, arguments):
        return arguments.symvars[self.idx - 1]


@dataclass
class MetavarAST(Variable):
    def to_python_ast(self):
        return ast.Name(id=self.sym)

    def _replace_with_substitute(self, arguments):
        return arguments.metavars[self.idx]


@dataclass
class ChoicevarAST(Variable):
    def to_python_ast(self):
        return ast.Name(id=self.sym)

    def _replace_with_substitute(self, arguments):
        return SpliceAST(arguments.choicevars[self.idx])


@dataclass
class AbstractionCallAST(PythonAST):
    tag: str
    args: List[PythonAST]
    handle: uuid.UUID

    def to_ns_s_exp(self, config):
        return SExpression(self.tag, [x.to_ns_s_exp(config) for x in self.args])

    def to_python_ast(self):
        raise RuntimeError("cannot convert abstraction call to python")

    def map(self, fn):
        return fn(
            AbstractionCallAST(self.tag, [x.map(fn) for x in self.args], self.handle)
        )

    def _collect_abstraction_calls(self, result):
        result[self.handle] = self
        return super()._collect_abstraction_calls(result)

    def _replace_abstraction_calls(self, handle_to_replacement):
        if self.handle in handle_to_replacement:
            return handle_to_replacement[self.handle]
        # pylint: disable=protected-access
        return self.map(
            lambda x: (
                x
                if isinstance(x, AbstractionCallAST) and x.tag == self.tag
                else x._replace_abstraction_calls(handle_to_replacement)
            )
        )


@dataclass
class SliceElementAST(PythonAST):
    content: PythonAST

    @classmethod
    def of(cls, x):
        if isinstance(x, SliceElementAST):
            return x
        return SliceElementAST(x)

    def to_ns_s_exp(self, config):
        # should not be necessary; since we have the assertion
        # but pylint is not smart enough to figure that out
        # pylint: disable=no-member
        content = self.content
        if isinstance(self.content, StarrableElementAST):
            # safe because it is not actually legal to have a starred element
            # in a slice
            content = content.content
        assert isinstance(content, (NodeAST, AbstractionCallAST, Variable)), content
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

    def substitute(self, arguments):
        return SliceElementAST(self.content.substitute(arguments))

    def map(self, fn):
        return fn(SliceElementAST(self.content.map(fn)))


@dataclass
class StarrableElementAST(PythonAST):
    content: PythonAST

    def to_ns_s_exp(self, config):
        # pylint: disable=no-member
        assert isinstance(
            self.content, (NodeAST, AbstractionCallAST, Variable)
        ), self.content
        if isinstance(self.content, NodeAST) and self.content.typ is ast.Starred:
            return SExpression("_starred_starred", [self.content.to_ns_s_exp(config)])
        return SExpression("_starred_content", [self.content.to_ns_s_exp(config)])

    def to_python_ast(self):
        return self.content.to_python_ast()

    def substitute(self, arguments):
        return StarrableElementAST(self.content.substitute(arguments))

    def map(self, fn):
        return fn(StarrableElementAST(self.content.map(fn)))


@dataclass
class SpliceAST(PythonAST):
    content: Union[SequenceAST, AbstractionCallAST]

    def __post_init__(self):
        assert isinstance(
            self.content, (SequenceAST, AbstractionCallAST, Variable)
        ), self.content

    def to_ns_s_exp(self, config):
        return SExpression("/splice", [self.content.to_ns_s_exp(config)])

    def to_python_ast(self):
        return Splice(self.content.to_python_ast())

    def map(self, fn):
        return fn(SpliceAST(self.content.map(fn)))
