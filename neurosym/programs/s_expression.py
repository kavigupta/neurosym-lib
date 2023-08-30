from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True, eq=True)
class SExpression:
    symbol: str
    children: Tuple["SExpression"]

    @property
    def postorder(self):
        for x in self.children:
            yield from x.postorder
        yield self


@dataclass
class InitializedSExpression:
    symbol: str
    children: Tuple["InitializedSExpression"]
    # state includes things related to the execution of the program,
    # e.g. weights of a neural network
    state: Dict[str, object]

    def all_state_values(self):
        yield from self.state.values()
        for child in self.children:
            yield from child.all_state_values()
