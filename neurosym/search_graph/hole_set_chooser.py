from abc import ABC, abstractmethod
from typing import List

from ..programs.hole import Hole, all_holes
from ..programs.s_expression import SExpression


class HoleSetChooser(ABC):
    @abstractmethod
    def choose_hole_sets(self, node: SExpression) -> List[List[Hole]]:
        """
        Returns a list of sets of holes, where each set of holes will be expanded
        simultaneously
        """


class ChooseFirst(HoleSetChooser):
    def choose_hole_sets(self, node: SExpression) -> List[List[Hole]]:
        result = []
        for hole in all_holes(node):
            result.append([hole])
            break
        return result


class ChooseAll(HoleSetChooser):
    def choose_hole_sets(self, node: SExpression) -> List[List[Hole]]:
        result = []
        for hole in all_holes(node):
            result.append([hole])
        return result
