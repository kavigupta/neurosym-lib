from types import NoneType
from typing import List, Union

from neurosym.types.type import (
    ArrowType,
    AtomicType,
    FilteredTypeVariable,
    ListType,
    TensorType,
    Type,
    TypeVariable,
)
from neurosym.types.type_signature import FunctionTypeSignature

SPECIAL_CHARS = ["{", "}", "[", "]", "(", ")", "->", ","]


class TypeDefiner:
    """
    A class that facilitates in parsing type strings, allowing for the definition of
        variables and filters that can be used to parse type strings.

    :param env: A dictionary that maps type variable names to objects, which
        can be types, sizes, etc.
    """

    def __init__(self, **env):
        self.env = env
        self.filters = {}

    def __call__(self, type_str: str) -> Type:
        """
        Parse a type string into a type.

        :param type_str: The type string to parse
        """
        return parse_type(type_str, self)

    def sig(self, type_str: str) -> FunctionTypeSignature:
        """
        Parse a type string into a function type signature. This is
        the same as :py:meth:`__call__`, but returns a :py:class:`FunctionTypeSignature`
        """
        typ = self(type_str)
        return FunctionTypeSignature(list(typ.input_type), typ.output_type)

    def typedef(self, key: str, type_str: str):
        """
        Define a type alias, which can be used to look up types later.

        E.g., ``typedef("fL", "{f, L}")`` defines a type alias ``$fL`` that can be used
        in other type strings.
        """
        if key[0] == "$":
            key = key[1:]
        self.env[key] = self(type_str)

    def filtered_type_variable(self, key, type_filter):
        """
        Set up a type variable with a filter. The type variable will be prefixed with a '%'
        instead of a '#'. The filter should be a function that takes a type and returns
        whether or not the variable can be instantiated with that type.
        """
        self.filters[key] = type_filter

    def lookup_type(self, key: str):
        """
        Look up a type definition.
        """
        return self.env[key]

    def lookup_filter(self, key: str):
        """
        Look up a filter definition.
        """
        return self.filters[key]


def render_type(t: Type) -> str:
    """
    Render a type into a human-readable string. Inverse of ``parse_type``.

    :param t: The type to render
    """
    if isinstance(t, AtomicType):
        return t.name
    if isinstance(t, TypeVariable):
        return "#" + t.name
    if isinstance(t, FilteredTypeVariable):
        return "%" + t.name
    if isinstance(t, TensorType):
        return "{" + ", ".join([render_type(t.dtype), *map(str, t.shape)]) + "}"
    if isinstance(t, ListType):
        return "[" + render_type(t.element_type) + "]"
    if isinstance(t, ArrowType):
        if len(t.input_type) == 1 and not isinstance(t.input_type[0], ArrowType):
            return render_type(t.input_type[0]) + " -> " + render_type(t.output_type)
        return (
            "("
            + ", ".join(map(render_type, t.input_type))
            + ") -> "
            + render_type(t.output_type)
        )
    raise NotImplementedError(f"Unknown type {t}")


def _parse_type_from_buf(reversed_buf, env: TypeDefiner):
    assert isinstance(env, TypeDefiner)
    first_tok = reversed_buf.pop()
    if first_tok.isnumeric():
        return int(first_tok)
    if first_tok.startswith("$"):
        return env.lookup_type(first_tok[1:])
    if first_tok == "{":
        internal_type = _parse_type_from_buf(reversed_buf, env)
        shape = []
        while True:
            tok = reversed_buf.pop()
            if tok == "}":
                break
            assert tok == ",", f"Expected ',' but got {tok}"
            size = _parse_type_from_buf(reversed_buf, env)
            shape.append(size)
        return TensorType(internal_type, tuple(shape))
    if first_tok == "[":
        internal_type = _parse_type_from_buf_multi(reversed_buf, env)
        close_bracket = reversed_buf.pop()
        assert close_bracket == "]", f"Expected ']' but got {close_bracket}"
        return ListType(internal_type)
    if first_tok == "(":
        input_types = []
        while True:
            if reversed_buf[-1] == ")":
                reversed_buf.pop()
                break
            input_types.append(_parse_type_from_buf_multi(reversed_buf, env))
            tok = reversed_buf.pop()
            if tok == ")":
                break
            assert tok == ",", f"Expected ',' but got {tok}"
        tok = reversed_buf.pop()
        assert tok == "->", f"Expected '->' but got {tok}"
        output_type = _parse_type_from_buf_multi(reversed_buf, env)
        return ArrowType(tuple(input_types), output_type)
    if first_tok.startswith("#"):
        return TypeVariable(first_tok[1:])
    if first_tok.startswith("%"):
        name = first_tok[1:]
        return FilteredTypeVariable(name, type_filter=env.lookup_filter(name))
    return AtomicType(first_tok)


def _parse_type_from_buf_multi(reversed_buf, env):
    t_head = _parse_type_from_buf(reversed_buf, env)
    if not reversed_buf:
        return t_head
    if reversed_buf and reversed_buf[-1] != "->":
        return t_head
    reversed_buf.pop()
    t_tail = _parse_type_from_buf_multi(reversed_buf, env)
    return ArrowType((t_head,), t_tail)


def lex_type(s: str) -> List[str]:
    """
    Lex a type string into tokens.
    """
    buf = []
    for c in s:
        if c in SPECIAL_CHARS:
            buf.append(c)
        elif c == " ":
            buf.append("")
        else:
            if len(buf) > 0 and buf[-1] not in SPECIAL_CHARS:
                buf[-1] += c
            else:
                buf.append(c)
    return [tok for tok in buf if tok != ""]


def parse_type(s, env: Union[TypeDefiner, NoneType] = None) -> Type:
    """
    Parse the given string into a type. The string should be in the format of the
    type string representation. The type string representation is a string that
    represents a type in a human-readable format.

    See the documentation for each Type subclass for more information on the
    type string representation for that type. A few edge cases for the ``ArrowType``
    are worth mentioning:

    If the input type is a single type, the parentheses are optional, unless the
    input type is another ``ArrowType``. So the parentheses are optional in the
    following cases

        .. code-block:: python

            (a) -> b
            ([a]) -> b
            ({a, 2}) -> b
            ([a -> b]) -> c

    but are required in the following cases

        .. code-block:: python

            (a, b) -> c
            (a -> b) -> c

    :param s: The string to parse
    :param env: The environment to use for looking up types and filters
    """
    if env is None:
        env = TypeDefiner()
    assert isinstance(env, TypeDefiner)
    buf = lex_type(s)[::-1]
    t = _parse_type_from_buf_multi(buf, env)
    assert buf == [], f"Extra tokens {buf[::-1]}"
    return t
