from neurosym.dsl.dsl import DSL
from neurosym.programs.hole import Hole
from neurosym.programs.s_expression import SExpression
from neurosym.search_graph.dsl_search_graph import DSLSearchGraph
from neurosym.search_graph.hole_set_chooser import ChooseFirst
from neurosym.search_graph.metadata_computer import NoMetadataComputer
from neurosym.search_graph.search_graph import SearchGraph
from neurosym.search_graph.search_graph_transformer import (
    FilterEdgesGraph,
    LimitEdgesGraph,
)
from neurosym.types.type import ArrowType, AtomicType, ListType, TensorType, Type
from neurosym.utils.logging import log


class FilterUnexpandableNodes(FilterEdgesGraph):
    """
    Represents a search graph that removes nodes that cannot be expanded.

    Removes edges to partial programs that will be too deep.
    """

    def __init__(self, graph: SearchGraph, max_depth: int):
        super().__init__(graph=graph)
        self.max_depth = max_depth

    def depth2go(self, node: SExpression) -> int:
        """
        Calculates a lower bound estimate of the depth2go. For a partial program,
        this is approximated by the minimum number of replacement nodes needed for
        each type of hole.

        :param node: The node to estimate depth2go for.
        :returns height: The minimum possible height of the node if partial programs
            were filled.
        """
        height = 1
        if isinstance(node, Hole):
            # @TODO: Better heuristic than this.
            match node.twe.typ:
                case ArrowType(input_type, _):
                    height += len(input_type)
                    # List type's require atleast two nodes to be fully symbolic.
                    height += bool(isinstance(input_type, ListType))
                    return height
                case TensorType(_):
                    height += 1
                    return height
                case AtomicType(_):
                    height += 1
                    return height
                case unk:
                    log(
                        f"Unknown Hole type {unk} encountered in depth2go. Assuming depth 1."
                    )
                    height += 1
                    return height

        max_child_height = 0
        for child in node.children:
            max_child_height = max(max_child_height, self.depth2go(child))

        height += max_child_height
        return height

    def include_edge(self, s, t):
        return self.depth2go(t.program) <= self.max_depth


def near_graph(
    dsl: DSL,
    root_type: Type,
    *,
    max_depth=1000,
    max_num_edges=100,
    is_goal=lambda x: True,
) -> SearchGraph:
    """
    Creates a search graph for the NEAR DSL.

    :param dsl: DSL to create the search graph for
    :param root_type: Type of the root node
    :param max_depth: Maximum depth for NEAR graph.
        Defaults to a really large number (1000).
    :param max_num_edges: Maximum number of edges for each node in the graph
    :param is_goal: Goal predicate
    """
    graph = DSLSearchGraph(
        dsl,
        root_type,
        ChooseFirst(),
        is_goal,
        NoMetadataComputer(),
    )
    graph = FilterUnexpandableNodes(graph, max_depth=max_depth)
    graph = LimitEdgesGraph(graph, max_num_edges)
    return graph
