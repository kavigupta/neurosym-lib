from abc import ABC, abstractmethod
from typing import List

from ..programs.hole import Hole, _all_holes
from ..programs.s_expression import SExpression


class HoleSetChooser(ABC):
    """
    Utility for the DSL search graph, which tells it how to choose sets of holes to expand.
    """

    @abstractmethod
    def choose_hole_sets(self, node: SExpression) -> List[List[Hole]]:
        """
        Returns a list of sets of holes, where each set of holes will be expanded
        simultaneously. The order of the sets of holes is the order in which they
        will be expanded.
        """


class ChooseFirst(HoleSetChooser):
    """
    Hole set chooser that chooses the first hole it finds.
    """

    def choose_hole_sets(self, node: SExpression) -> List[List[Hole]]:
        result = []
        for hole in _all_holes(node):
            result.append([hole])
            break
        return result


class ChooseAll(HoleSetChooser):
    """
    Hole set chooser that chooses all holes.
    """

    def choose_hole_sets(self, node: SExpression) -> List[List[Hole]]:
        result = []
        for hole in _all_holes(node):
            result.append([hole])
        return result
