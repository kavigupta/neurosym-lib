import itertools
from types import NoneType
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


class DSLSearchGraph(SearchGraph[SExpression]):
    """
    Represents a search graph where nodes are ns.SExpression objects with holes
    in them, and edges are expansions of those holes.

    :param dsl: DSL to use for expanding holes
    :param target_type: Type of the SExpressions we are searching for
    :param hole_set_chooser: Chooser for sets of holes to expand
    :param test_predicate: Predicate for goal nodes
    :param metadata_computer: Computer for metadata for nodes
    :param compute_cost: Function to compute the cost of a node
    :param skip_ahead: Whether to skip ahead in the search graph whenever
        a node coming up has only one possible expansion. This can be useful
        when the cost of whatever downstream task you have for a partial
        node is more expensive than the cost of expanding it (possibly) twice.
    """

    def __init__(
        self,
        dsl: DSL,
        target_type: Type,
        hole_set_chooser: HoleSetChooser,
        test_predicate: Callable[[SExpression], bool],
        metadata_computer: MetadataComputer,
        compute_cost: Callable[[DSLSearchNode], float] | NoneType = None,
        skip_ahead=False,
    ):
        self.dsl = dsl
        self.target_type = target_type
        self.hole_set_chooser = hole_set_chooser
        self.test_predicate = test_predicate
        self.metadata_computer = metadata_computer
        self.compute_cost = compute_cost
        self.skip_ahead = skip_ahead

    def initial_node(self):
        return DSLSearchNode(
            Hole.of(TypeWithEnvironment(self.target_type, Environment.empty())),
            self.dsl,
            self.metadata_computer.for_initial_node(),
        )

    def expand_node(self, node):
        assert isinstance(node, DSLSearchNode)
        if not self.skip_ahead:
            yield from self._direct_expand_node(node)
        for expanded_node in self._direct_expand_node(node):
            yield self._maximally_expanded_node(expanded_node)

    def _direct_expand_node(self, node):
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
                    self.dsl,
                    self.metadata_computer.for_expanded_node(node, expanded_program),
                )

    def _maximally_expanded_node(self, node, depth=0):
        if self.is_goal_node(node) or depth >= 10:
            # always return goal nodes
            return node
        expansions = self._direct_expand_node(node)
        try:
            first_expansion = next(expansions)
        except StopIteration:
            # No expansions, so return the node as is
            return node
        try:
            next(expansions)
            # More than one expansion, so return the node as is
            return node
        except StopIteration:
            # Exactly one expansion, so try to expand it further
            return self._maximally_expanded_node(first_expansion, depth + 1)

    def is_goal_node(self, node):
        if any(True for _ in _all_holes(node.program)):
            return False
        return self.test_predicate(node)

    def cost(self, node) -> float:
        if self.compute_cost is None:
            raise NotImplementedError("Cost function not provided")
        return self.compute_cost(node)

    def finalize(self, node) -> SExpression:
        return node.program
