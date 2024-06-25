import itertools
from typing import Callable

from neurosym.search_graph.dsl_search_node import DSLSearchNode
from neurosym.search_graph.metadata_computer import MetadataComputer
from neurosym.types.type_with_environment import Environment, TypeWithEnvironment

from ..dsl.dsl import DSL
from ..programs.hole import Hole, _all_holes, _replace_holes
from ..programs.s_expression import SExpression
from ..types.type import Type
from .hole_set_chooser import HoleSetChooser
from .search_graph import SearchGraph


class DSLSearchGraph(SearchGraph):
    """
    Represents a search graph where nodes are ns.SExpression objects with holes
    in them, and edges are expansions of those holes.

    :param dsl: DSL to use for expanding holes
    :param target_type: Type of the SExpressions we are searching for
    :param hole_set_chooser: Chooser for sets of holes to expand
    :param test_predicate: Predicate for goal nodes
    :param metadata_computer: Computer for metadata for nodes
    """

    def __init__(
        self,
        dsl: DSL,
        target_type: Type,
        hole_set_chooser: HoleSetChooser,
        test_predicate: Callable[[SExpression], bool],
        metadata_computer: MetadataComputer,
    ):
        self.dsl = dsl
        self.target_type = target_type
        self.hole_set_chooser = hole_set_chooser
        self.test_predicate = test_predicate
        self.metadata_computer = metadata_computer

    def initial_node(self):
        return DSLSearchNode(
            Hole.of(TypeWithEnvironment(self.target_type, Environment.empty())),
            self.metadata_computer.for_initial_node(),
        )

    def expand_node(self, node):
        hole_sets = self.hole_set_chooser.choose_hole_sets(node.program)
        relevant_holes = sorted(set(h for hs in hole_sets for h in hs))
        # relevant_productions[hole_idx] : list of expansions for holes[hole_idx]
        relevant_productions = {}
        for hole in relevant_holes:
            relevant_productions[hole] = self.dsl.expansions_for_type(hole.twe)
        for hole_set in hole_sets:
            hole_set = sorted(hole_set)
            for hole_replacements in itertools.product(
                *[relevant_productions[h] for h in hole_set]
            ):
                # hole_replacements : list of SExpressions, of length len(holes)
                expanded_program = _replace_holes(
                    node.program, hole_set, hole_replacements
                )
                yield DSLSearchNode(
                    expanded_program,
                    self.metadata_computer.for_expanded_node(node, expanded_program),
                )

    def is_goal_node(self, node):
        if any(True for _ in _all_holes(node.program)):
            return False
        return self.test_predicate(node)
