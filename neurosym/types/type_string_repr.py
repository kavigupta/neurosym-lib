from types import NoneType
from typing import Union

from neurosym.types.type import (
    ArrowType,
    AtomicType,
    FilteredTypeVariable,
    ListType,
    TensorType,
    TypeVariable,
)
from neurosym.types.type_signature import FunctionTypeSignature

SPECIAL_CHARS = ["{", "}", "[", "]", "(", ")", "->", ","]


class TypeDefiner:
    def __init__(self, **env):
        self.env = env
        self.filters = {}

    def __call__(self, type_str):
        return parse_type(type_str, self)

    def sig(self, type_str):
        typ = self(type_str)
        return FunctionTypeSignature(list(typ.input_type), typ.output_type)

    def typedef(self, key, type_str):
        self.env[key] = self(type_str)

    def filtered_type_variable(self, key, type_filter):
        self.filters[key] = type_filter

    def lookup_type(self, key):
        return self.env[key]

    def lookup_filter(self, key):
        return self.filters[key]


def render_type(t):
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


def parse_type_from_buf(reversed_buf, env: TypeDefiner):
    assert isinstance(env, TypeDefiner)
    first_tok = reversed_buf.pop()
    if first_tok.isnumeric():
        return int(first_tok)
    if first_tok.startswith("$"):
        return env.lookup_type(first_tok[1:])
    if first_tok == "{":
        internal_type = parse_type_from_buf(reversed_buf, env)
        shape = []
        while True:
            tok = reversed_buf.pop()
            if tok == "}":
                break
            assert tok == ",", f"Expected ',' but got {tok}"
            size = parse_type_from_buf(reversed_buf, env)
            shape.append(size)
        return TensorType(internal_type, tuple(shape))
    if first_tok == "[":
        internal_type = parse_type_from_buf_multi(reversed_buf, env)
        close_bracket = reversed_buf.pop()
        assert close_bracket == "]", f"Expected ']' but got {close_bracket}"
        return ListType(internal_type)
    if first_tok == "(":
        input_types = []
        while True:
            if reversed_buf[-1] == ")":
                reversed_buf.pop()
                break
            input_types.append(parse_type_from_buf_multi(reversed_buf, env))
            tok = reversed_buf.pop()
            if tok == ")":
                break
            assert tok == ",", f"Expected ',' but got {tok}"
        tok = reversed_buf.pop()
        assert tok == "->", f"Expected '->' but got {tok}"
        output_type = parse_type_from_buf_multi(reversed_buf, env)
        return ArrowType(tuple(input_types), output_type)
    if first_tok.startswith("#"):
        return TypeVariable(first_tok[1:])
    if first_tok.startswith("%"):
        name = first_tok[1:]
        return FilteredTypeVariable(name, type_filter=env.lookup_filter(name))
    return AtomicType(first_tok)


def parse_type_from_buf_multi(reversed_buf, env):
    t_head = parse_type_from_buf(reversed_buf, env)
    if not reversed_buf:
        return t_head
    if reversed_buf and reversed_buf[-1] != "->":
        return t_head
    reversed_buf.pop()
    t_tail = parse_type_from_buf_multi(reversed_buf, env)
    return ArrowType((t_head,), t_tail)


def lex(s):
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


def parse_type(s, env: Union[TypeDefiner, NoneType] = None):
    if env is None:
        env = TypeDefiner()
    assert isinstance(env, TypeDefiner)
    buf = lex(s)[::-1]
    t = parse_type_from_buf_multi(buf, env)
    assert buf == [], f"Extra tokens {buf[::-1]}"
    return t
