import re
from typing import Set

from neurosym.programs.s_expression import SExpression


def variable_indices(s_exp: SExpression) -> Set[int]:
    """
    Extract the set of environment indices referenced by variable
    productions in the given s-expression.

    :param s_exp: The s-expression to scan.
    :return: A set of integer environment indices.
    """
    result = set()
    match = re.match(r"^\$(\d+)_\d+$", s_exp.symbol)
    if match:
        result.add(int(match.group(1)))
    for child in s_exp.children:
        result.update(variable_indices(child))
    return result


def remove_shield_productions(s_exp: SExpression) -> SExpression:
    """
    Remove all shield productions from an s-expression, adjusting
    variable indices to account for the removed environment entries.

    Processes bottom-up: inner shields are removed first, then outer ones.

    :param s_exp: The s-expression to sanitize.
    :return: A new s-expression with all shield productions removed.
    """
    # Recursively process children first
    children = tuple(remove_shield_productions(child) for child in s_exp.children)
    s_exp = SExpression(s_exp.symbol, children)

    # Check if this is a shield production
    match = re.match(r"^shield(\d+)$", s_exp.symbol)
    if match:
        k = int(match.group(1))
        assert len(s_exp.children) == 1
        return _shift_variables(s_exp.children[0], k)

    return s_exp


def _shift_variables(s_exp: SExpression, k: int) -> SExpression:
    """
    Shift all variable references with env index >= k up by 1.
    """
    match = re.match(r"^\$(\d+)_(\d+)$", s_exp.symbol)
    if match:
        n = int(match.group(1))
        type_id = match.group(2)
        if n >= k:
            return SExpression(f"${n + 1}_{type_id}", s_exp.children)
        return s_exp

    children = tuple(_shift_variables(child, k) for child in s_exp.children)
    return SExpression(s_exp.symbol, children)
