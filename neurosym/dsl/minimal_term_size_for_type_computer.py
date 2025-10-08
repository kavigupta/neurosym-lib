from typing import Dict

from neurosym.types.type_with_environment import TypeWithEnvironment
from neurosym.utils.documentation import internal_only


@internal_only
class MinimalTermSizeForTypeComputer:

    def __init__(self, dsl, symbol_costs: Dict[str, int]):
        self.dsl = dsl
        self._value = {}
        self._at_most = {}
        self.symbol_costs = symbol_costs

    @internal_only
    def compute(self, typ: TypeWithEnvironment):
        while True:
            updated, value = self._run_full_update(typ)
            if not updated:
                return value

    def _run_full_update(self, typ: TypeWithEnvironment):
        fringe = [typ]
        seen = set()
        updated = False
        while fringe:
            current = fringe.pop()
            if current in seen:
                continue
            seen.add(current)
            updated_this, _ = self._run_update(current, fringe)
            updated |= updated_this
        if not updated:
            self._value[typ] = self._at_most.get(typ, float("inf"))
        return updated, self._value.get(typ, float("inf"))

    def _run_update(self, typ: TypeWithEnvironment, fringe):
        if typ in self._value:
            return False, self._value[typ]
        best_size = self._at_most.get(typ, float("inf"))
        for expansion in self.dsl.expansions_for_type(typ):
            size_so_far = self.symbol_costs.get(expansion.symbol, 1)
            children_types = [child.twe for child in expansion.children]
            for child_type in children_types:
                fringe.append(child_type)
                child_size = self._at_most.get(child_type, float("inf"))
                size_so_far += child_size
            best_size = min(best_size, size_so_far)
        updated = False
        if self._at_most.get(typ, float("inf")) != best_size:
            updated = True
            self._at_most[typ] = best_size
        return updated, best_size
