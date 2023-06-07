from abc import ABC, abstractmethod
import itertools
from typing import Callable


from ..dsl.dsl import DSL
from ..programs.hole import Hole, all_holes, replace_holes
from ..programs.s_expression import SExpression
from ..types.type import Type

from .hole_set_chooser import HoleSetChooser
from .search_graph import SearchGraph


class DSLSearchGraph(SearchGraph, ABC):
    """
    Represents a search graph where nodes are SExpressions with holes in them, and edges are
    expansions of those holes.

    :param dsl: DSL to use for expanding holes
    :param target_type: Type of the SExpressions we are searching for
    """

    def __init__(
        self,
        dsl: DSL,
        target_type: Type,
        hole_set_chooser: HoleSetChooser,
        test_predicate: Callable[[SExpression], bool],
    ):
        self.dsl = dsl
        self.target_type = target_type
        self.hole_set_chooser = hole_set_chooser
        self.test_predicate = test_predicate

    def initial_node(self):
        return Hole.of(self.target_type)

    def expand_node(self, node):
        hole_sets = self.hole_set_chooser.choose_hole_sets(node)
        relevant_holes = sorted(set(h for hs in hole_sets for h in hs))
        # relevant_productions[hole_idx] : list of expansions for holes[hole_idx]
        relevant_productions = {}
        for hole in relevant_holes:
            relevant_productions[hole] = self.dsl.expansions_for_type(hole.type)
        for hole_set in hole_sets:
            hole_set = sorted(hole_set)
            for hole_replacements in itertools.product(
                *[relevant_productions[h] for h in hole_set]
            ):
                # hole_replacements : list of SExpressions, of length len(holes)
                yield replace_holes(node, hole_set, hole_replacements)

    def is_goal_node(self, node):
        if any(True for _ in all_holes(node)):
            return False
        return self.test_predicate(node)
