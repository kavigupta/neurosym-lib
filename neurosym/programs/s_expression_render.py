from typing import List, Set
from s_expression_parser import parse, ParserConfig, Pair, nil, Renderer

from neurosym.programs.s_expression import SExpression


def to_pair(s_exp: SExpression, *, for_stitch: bool) -> Pair:
    """
    Convert an SExpression to a Pair.

    If we are exporting for stitch and the SExpression is a leaf,
        it will be converted to a string with the prefix "leaf-".
        This is because stitch does not distinguish `(f)` from `f`.

    Args:
        s_exp: The SExpression to convert.
        for_stitch: Whether the Pair is being converted for use with stitch.
    Returns:
        The Pair representing the SExpression.
    """
    if hasattr(s_exp, "__to_pair__"):
        return s_exp.__to_pair__(for_stitch=for_stitch)
    assert isinstance(s_exp, SExpression), f"Expected SExpression, got {s_exp}"
    if for_stitch and not s_exp.children:
        return "leaf-" + s_exp.symbol
    elements = [s_exp.symbol] + [
        to_pair(x, for_stitch=for_stitch) for x in s_exp.children
    ]
    result = nil
    for element in reversed(elements):
        result = Pair(element, result)
    return result


def from_pair(pair: Pair, should_not_be_leaf: Set[str]) -> SExpression:
    """
    Convert a Pair to an SExpression.

    Args:
        pair: The Pair to convert.
        should_not_be_leaf: A set of symbols that should not be converted to leaves.
            Instead, they should be converted to SExpressions with no children.

    Returns:
        The SExpression representing the Pair.
    """
    if isinstance(pair, str):
        if pair.startswith("leaf-"):
            return SExpression(pair[5:], [])
        if pair in should_not_be_leaf:
            return SExpression(pair, [])
        return pair
    assert isinstance(pair, Pair), f"Expected pair, got {pair}"
    elements = []
    while pair is not nil:
        car = pair.car
        if not elements:
            assert isinstance(car, str), f"Expected string, got {pair.car}"
        else:
            car = from_pair(car, should_not_be_leaf)
        elements.append(car)
        pair = pair.cdr
    head, *tail = elements
    return SExpression(head, tail)


def render_s_expression(s_exp: SExpression, for_stitch: bool) -> str:
    """
    Render an SExpression as a string.

    Args:
        s_exp: The SExpression to render.
        for_stitch: Whether the Pair is being converted for use with stitch.

    Returns:
        The string representing the SExpression.
    """
    return Renderer(columns=float("inf")).render(to_pair(s_exp, for_stitch=for_stitch))


def parse_s_expression(s: str, should_not_be_leaf: Set[str]) -> SExpression:
    """
    Parse a string into an SExpression.

    Args:
        s: The string to parse.
        should_not_be_leaf: A set of symbols that should not be converted to leaves.
            Instead, they should be converted to SExpressions with no children.

    Returns:
        The SExpression representing the string.
    """
    [pair] = parse(s, ParserConfig((), dots_are_cons=False))
    return from_pair(pair, should_not_be_leaf=should_not_be_leaf)


def symbols(s: SExpression) -> List[str]:
    return [s.symbol] + [sym for x in s.children for sym in symbols(x)]
