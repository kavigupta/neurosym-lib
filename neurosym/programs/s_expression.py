from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True, eq=True)
class SExpression:
    symbol: str
    children: Tuple["SExpression"]


@dataclass
class InitializedSExpression:
    symbol: str
    children: Tuple["InitializedSExpression"]
    state: Dict[str, object]
