from dataclasses import dataclass
from typing import Dict

from neurosym.types.type_with_environment import TypeWithEnvironment
from neurosym.utils.documentation import internal_only


@dataclass
class _AtLeast:
    value: int

    def __add__(self, other):
        if isinstance(other, _AtLeast):
            return _AtLeast(self.value + other.value)
        return _AtLeast(self.value + other)

    def __radd__(self, other):
        return self + other


def _definitely_less_than_or_eq(value, limit):
    if isinstance(value, _AtLeast):
        return False
    return value <= limit


def _definitely_greater_than_or_eq(value, limit):
    if isinstance(value, _AtLeast):
        return value.value >= limit
    return value >= limit


def _compute_min(a, b):
    if isinstance(a, _AtLeast) and isinstance(b, _AtLeast):
        return _AtLeast(min(a.value, b.value))
    if isinstance(b, _AtLeast):
        # pylint: disable=arguments-out-of-order
        return _compute_min(b, a)
    if not isinstance(a, _AtLeast):
        return min(a, b)
    if a.value >= b:
        return b
    raise ValueError(f"Cannot compare {a} and {b}")


@internal_only
class MinimalTermSizeForTypeComputer:
    def __init__(self, dsl):
        self.dsl = dsl
        self._by_type = {}

    @internal_only
    def compute(self, typ: TypeWithEnvironment, symbol_costs: Dict[str, int]):
        # iterative deepening search
        depth_limit = 1
        while True:
            result = self._compute_with_depth(typ, symbol_costs, depth_limit)
            if _definitely_less_than_or_eq(result, depth_limit):
                return result
            if isinstance(result, _AtLeast) and result.value == float("inf"):
                return float("inf")
            depth_limit *= 2

    def _compute_with_depth(
        self, typ: TypeWithEnvironment, symbol_costs: Dict[str, int], depth_limit: int
    ):
        if depth_limit <= 0:
            return _AtLeast(0)
        if typ in self._by_type:
            if _definitely_less_than_or_eq(self._by_type[typ], depth_limit):
                return self._by_type[typ]
            if _definitely_greater_than_or_eq(self._by_type[typ], depth_limit):
                return self._by_type[typ]
        best_size = _AtLeast(float("inf"))
        for expansion in self.dsl.expansions_for_type(typ):
            size_so_far = symbol_costs.get(expansion.symbol, 1)
            children_types = [child.twe for child in expansion.children]
            for child_type in children_types:
                child_size = self._compute_with_depth(
                    child_type, symbol_costs, depth_limit - size_so_far
                )
                size_so_far += child_size
                if isinstance(size_so_far, _AtLeast):
                    break
                if size_so_far >= depth_limit:
                    size_so_far = _AtLeast(size_so_far)
                    break
            best_size = _compute_min(best_size, size_so_far)
        self._by_type[typ] = best_size
        return best_size
