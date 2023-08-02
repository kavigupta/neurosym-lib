from neurosym.programs.hole import all_holes
from neurosym.search_graph.metadata_computer import MetadataComputer


class NearMetadataComputer(MetadataComputer):
    """
    Represents a metadata computer that contains metadata
    for the NEAR algorithm.
    """

    def for_initial_node(self):
        return dict(depth=0, cost=0)

    def for_expanded_node(self, node, expanded_node):
        return dict(depth=node["depth"] + 1, cost=len(list(all_holes(expanded_node))))
