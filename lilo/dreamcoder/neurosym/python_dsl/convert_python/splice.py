from dataclasses import dataclass
from typing import List

from neurosym.utils.documentation import internal_only


@internal_only
@dataclass
class Splice:
    """
    Represents an object that, if placed in a list, should be replaced by the
    elements of the target list.

    E.g., you can think of [A, Splice([B, C]), D] as equivalent to [A, B, C, D].
    """

    target: List[object]

    def __post_init__(self):
        assert isinstance(self.target, list), self.target
