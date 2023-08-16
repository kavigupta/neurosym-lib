from neurosym.types.type import ArrowType, AtomicType, ListType, TensorType

SPECIAL_CHARS = ["{", "}", "[", "]", "(", ")", "->", ","]


class TypeDefiner:
    def __init__(self, **env):
        self.env = env

    def __call__(self, type_str):
        return parse_type(type_str, self.env)

    def typedef(self, key, type_str):
        self.env[key] = self(type_str)


def render_type(t):
    if isinstance(t, AtomicType):
        return t.name
    elif isinstance(t, TensorType):
        return "{" + ", ".join([render_type(t.dtype), *map(str, t.shape)]) + "}"
    elif isinstance(t, ListType):
        return "[" + render_type(t.element_type) + "]"
    elif isinstance(t, ArrowType):
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
            assert tok == ","
            size = parse_type_from_buf(reversed_buf, env)
            shape.append(size)
        return TensorType(internal_type, tuple(shape))
    elif first_tok == "[":
        internal_type = parse_type_from_buf(reversed_buf, env)
        assert reversed_buf.pop() == "]"
        return ListType(internal_type)
    elif first_tok == "(":
        input_types = []
        while True:
            input_types.append(parse_type_from_buf(reversed_buf, env))
            tok = reversed_buf.pop()
            if tok == ")":
                break
            assert tok == ","
        assert reversed_buf.pop() == "->"
        output_type = parse_type_from_buf(reversed_buf, env)
        return ArrowType(tuple(input_types), output_type)
    else:
        return AtomicType(first_tok)


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
    t = parse_type_from_buf(buf, env)
    assert len(buf) == 0
    return t
