from neurosym.dsl.dsl import DSL
from neurosym.search_graph.dsl_search_graph import DSLSearchGraph
from neurosym.search_graph.hole_set_chooser import ChooseFirst
from neurosym.search_graph.metadata_computer import NoMetadataComputer
from neurosym.search_graph.search_graph import SearchGraph
from neurosym.search_graph.search_graph_transformer import (
    FilterEdgesGraph,
    LimitEdgesGraph,
)
from neurosym.types.type import Type


class FilterUnexpandableNodes(FilterEdgesGraph):
    """
    Represents a search graph that removes nodes that cannot be expanded.
    """

    def include_edge(self, s, t):
        del s
        # TODO(AS) throw out nodes that cannot
        # be expanded because they would be too deep
        return True


def near_graph(
    dsl: DSL, root_type: Type, *, max_num_edges=100, is_goal=lambda x: True
) -> SearchGraph:
    """
    Creates a search graph for the NEAR DSL.

    :param dsl: DSL to create the search graph for
    :param root_type: Type of the root node
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
    graph = FilterUnexpandableNodes(graph)
    graph = LimitEdgesGraph(graph, max_num_edges)
    return graph
