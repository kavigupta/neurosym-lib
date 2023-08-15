from neurosym.types.type import ArrowType, AtomicType, ListType, TensorType

SPECIAL_CHARS = ["{", "}", "[", "]", "(", ")", "->", ","]


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


def parse_type_from_buf(reversed_buf):
    first_tok = reversed_buf.pop()
    if first_tok == "{":
        internal_type = parse_type_from_buf(reversed_buf)
        shape = []
        while True:
            tok = reversed_buf.pop()
            if tok == "}":
                break
            assert tok == ","
            tok = reversed_buf.pop()
            shape.append(int(tok))
        return TensorType(internal_type, tuple(shape))
    elif first_tok == "[":
        internal_type = parse_type_from_buf(reversed_buf)
        assert reversed_buf.pop() == "]"
        return ListType(internal_type)
    elif first_tok == "(":
        input_types = []
        while True:
            input_types.append(parse_type_from_buf(reversed_buf))
            tok = reversed_buf.pop()
            if tok == ")":
                break
            assert tok == ","
        assert reversed_buf.pop() == "->"
        output_type = parse_type_from_buf(reversed_buf)
        return ArrowType(tuple(input_types), output_type)
    else:
        return AtomicType(first_tok)


def lex(s):
    buf = []
    for c in s:
        if c in SPECIAL_CHARS:
            buf.append(c)
        elif c == " ":
            pass
        else:
            if len(buf) > 0 and buf[-1] not in SPECIAL_CHARS:
                buf[-1] += c
            else:
                buf.append(c)
    return buf


def parse_type(s):
    return parse_type_from_buf(lex(s)[::-1])
