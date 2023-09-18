from neurosym.types.type import (
    ArrowType,
    AtomicType,
    ListType,
    TensorType,
    TypeVariable,
)
from neurosym.types.type_signature import ConcreteTypeSignature

SPECIAL_CHARS = ["{", "}", "[", "]", "(", ")", "->", ","]


class TypeDefiner:
    def __init__(self, **env):
        self.env = env

    def __call__(self, type_str):
        return parse_type(type_str, self.env)

    def sig(self, type_str):
        typ = self(type_str)
        return ConcreteTypeSignature(list(typ.input_type), typ.output_type)

    def typedef(self, key, type_str):
        self.env[key] = self(type_str)


def render_type(t):
    if isinstance(t, AtomicType):
        return t.name
    if isinstance(t, TypeVariable):
        return "#" + t.name
    elif isinstance(t, TensorType):
        return "{" + ", ".join([render_type(t.dtype), *map(str, t.shape)]) + "}"
    elif isinstance(t, ListType):
        return "[" + render_type(t.element_type) + "]"
    elif isinstance(t, ArrowType):
        if len(t.input_type) == 1 and not isinstance(t.input_type[0], ArrowType):
            return render_type(t.input_type[0]) + " -> " + render_type(t.output_type)
        return (
            "("
            + ", ".join(map(render_type, t.input_type))
            + ") -> "
            + render_type(t.output_type)
        )
    else:
        raise NotImplementedError(f"Unknown type {t}")


def parse_type_from_buf(reversed_buf, env):
    first_tok = reversed_buf.pop()
    if first_tok.isnumeric():
        return int(first_tok)
    elif first_tok.startswith("$"):
        return env[first_tok[1:]]
    elif first_tok == "{":
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
    elif first_tok == "[":
        internal_type = parse_type_from_buf_multi(reversed_buf, env)
        close_bracket = reversed_buf.pop()
        assert close_bracket == "]", f"Expected ']' but got {close_bracket}"
        return ListType(internal_type)
    elif first_tok == "(":
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
    elif first_tok.startswith("#"):
        return TypeVariable(first_tok[1:])
    else:
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


def parse_type(s, env=None):
    if env is None:
        env = {}
    buf = lex(s)[::-1]
    t = parse_type_from_buf_multi(buf, env)
    assert buf == [], f"Extra tokens {buf[::-1]}"
    return t
