from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True, eq=True)
class SExpression:
    symbol: str
    children: Tuple["SExpression"]


@dataclass
class InitializedSExpression:
    symbol: str
    children: Tuple["InitializedSExpression"]
    # state includes things related to the execution of the program, e.g. weights of a neural network
    state: Dict[str, object]
