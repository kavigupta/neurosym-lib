from typing import Iterator, Set

from s_expression_parser import Pair, ParserConfig, Renderer, nil, parse

from neurosym.programs.s_expression import SExpression


def _to_pair(
    s_exp: SExpression, *, for_stitch: bool, allow_sexp_head: bool = False
) -> Pair:
    """
    Convert an SExpression to a Pair.

    If we are exporting for stitch and the SExpression is a leaf,
        it will be converted to a string with the prefix "leaf-".
        This is because stitch does not distinguish ``(f)`` from ``f``.

    Args:
        s_exp: The SExpression to convert.
        for_stitch: Whether the Pair is being converted for use with stitch.
    Returns:
        The Pair representing the SExpression.
    """
    if hasattr(s_exp, "__to_pair__"):
        return s_exp.__to_pair__(for_stitch=for_stitch)
    if isinstance(s_exp, str):
        return s_exp
    assert hasattr(s_exp, "children"), f"Expected SExpression, got {s_exp}"
    symbol = s_exp.symbol
    if isinstance(symbol, SExpression):
        if not allow_sexp_head:
            raise ValueError(
                "SExpression head is not allowed in this context. "
                "Set allow_sexp_head=True to allow it."
            )
        symbol = _to_pair(symbol, for_stitch=for_stitch, allow_sexp_head=True)
    if for_stitch and not s_exp.children:
        if not isinstance(symbol, str):
            raise ValueError(
                "Expected string symbol for leaf, got {symbol} of type {type(symbol)}"
            )
        if symbol.startswith("$"):
            return symbol
        return "leaf-" + symbol
    elements = [symbol] + [_to_pair(x, for_stitch=for_stitch) for x in s_exp.children]
    result = nil
    for element in reversed(elements):
        result = Pair(element, result)
    return result


def _from_pair(
    pair: Pair,
    should_not_be_leaf: Set[str],
    for_stitch: bool,
    allow_sexp_head: bool,
) -> SExpression:
    """
    Convert a Pair to an SExpression.

    Args:
        pair: The Pair to convert.
        should_not_be_leaf: A set of symbols that should not be converted to leaves.
            Instead, they should be converted to SExpressions with no children.
        for_stitch: Whether the Pair is being converted for use with stitch.
        allow_sexp_head: Whether we are allowing the head of the SExpression
            to be an SExpression. This is useful for parsing the head of a program
            as an SExpression, which is not standard.

    Returns:
        The SExpression representing the Pair.
    """
    if isinstance(pair, str):
        if for_stitch:
            if pair.startswith("leaf-"):
                return SExpression(pair[5:], ())
            if pair.startswith("$"):
                return SExpression(pair, ())
        if pair in should_not_be_leaf:
            return SExpression(pair, ())
        return pair
    assert isinstance(pair, Pair), f"Expected pair, got {pair}"
    elements = []
    while pair is not nil:
        car = pair.car
        if not elements:
            if allow_sexp_head:
                car = _from_pair(
                    car,
                    should_not_be_leaf=should_not_be_leaf,
                    for_stitch=for_stitch,
                    allow_sexp_head=allow_sexp_head,
                )
            else:
                assert isinstance(car, str), f"Expected string, got {pair.car}"
        else:
            car = _from_pair(
                car,
                should_not_be_leaf,
                for_stitch=for_stitch,
                allow_sexp_head=allow_sexp_head,
            )
        elements.append(car)
        pair = pair.cdr
    head, *tail = elements
    return SExpression(head, tuple(tail))


def render_s_expression(
    s_exp: SExpression, for_stitch: bool = False, allow_sexp_head: bool = False
) -> str:
    """
    Render an SExpression as a string.

    :param s_exp: The SExpression to render.
    :param for_stitch: Whether we are parsing this expression from ``stitch_core``,
        in which case we need to handle the "leaf-" prefix that we introduced
        in ``parse_s_expression``.
    :param allow_sexp_head: Whether we are allowing the head of the SExpression
        to be an SExpression. This is useful for parsing the head of a program
        as an SExpression, which is not standard.

    :returns: The string representing the SExpression.
    """
    return Renderer(columns=float("inf")).render(
        _to_pair(s_exp, for_stitch=for_stitch, allow_sexp_head=allow_sexp_head)
    )


def parse_s_expression(
    s: str,
    *,
    should_not_be_leaf: Set[str] = None,
    for_stitch: bool = False,
    allow_sexp_head: bool = False,
) -> SExpression:
    """
    Parse an SExpression from a string.

    :param s: The string to parse.
    :param should_not_be_leaf: A set of symbols that should not be converted to leaves.
        Instead, they should be converted to SExpressions with no children.
    :param for_stitch: Whether we are rendering this s-expression for ``stitch_core``,
        in which case we need to tag leaf nodes with "leaf-" prefix.
    :param allow_sexp_head: Whether we are allowing the head of the SExpression
        to be an SExpression. This is useful for parsing the head of a program
        as an SExpression, which is not standard.

    :returns: The SExpression representing the string.
    """
    if should_not_be_leaf is None:
        should_not_be_leaf = set()
    pairs = parse(s, ParserConfig((), dots_are_cons=False))
    if len(pairs) != 1:
        raise ValueError(f"Expected one expression, got {len(pairs)}")
    return _from_pair(
        pairs[0],
        should_not_be_leaf=should_not_be_leaf,
        for_stitch=for_stitch,
        allow_sexp_head=allow_sexp_head,
    )


def _symbols_for_program_gen(s: SExpression) -> Iterator[str]:
    """
    Produce a list of symbols in a program.
    """
    yield s.symbol
    for x in s.children:
        yield from _symbols_for_program_gen(x)


def symbols_for_program(s: SExpression) -> Set[str]:
    """
    Produce a set of symbols in a program.
    """
    return set(_symbols_for_program_gen(s))
