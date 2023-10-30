from dataclasses import dataclass, field
from typing import Dict

from neurosym.programs.s_expression import SExpression


@dataclass(unsafe_hash=True)
class DSLSearchNode:
    program: SExpression
    # metadata includes things related to the search graph, e.g., depth of the node
    metadata: Dict[str, object] = field(hash=False)
