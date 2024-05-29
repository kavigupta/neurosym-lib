from dataclasses import dataclass
from typing import List


@dataclass
class Splice:
    target: List[object]

    def __post_init__(self):
        assert isinstance(self.target, list), self.target
