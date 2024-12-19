from dataclasses import dataclass, field
from typing import Dict

from neurosym.dsl.dsl import DSL
from neurosym.programs.s_expression import SExpression


@dataclass(unsafe_hash=True)
class DSLSearchNode:
    """
    Search node for the ``DSLSearchGraph``.

    :param program: The program represented by this node.
    :param metadata: Metadata associated with this node. metadata includes
        things related to the search graph, e.g., depth of the node
    """

    program: SExpression
    dsl: DSL = field(hash=False)
    metadata: Dict[str, object] = field(hash=False)
