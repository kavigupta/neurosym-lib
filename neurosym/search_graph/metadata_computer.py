from abc import ABC, abstractmethod


class MetadataComputer(ABC):
    """
    Represents a way to compute metadata for a node in a search graph.
    """

    @abstractmethod
    def for_initial_node(self):
        """
        Return the metadata for the initial node in the search graph.
        """

    @abstractmethod
    def for_expanded_node(self, node, expanded_node):
        """
        Return the metadata for the expanded node in the search graph.
        """


class NoMetadataComputer(MetadataComputer):
    """
    Represents a metadata computer that returns empty metadata for all nodes.
    """

    def for_initial_node(self):
        return {}

    def for_expanded_node(self, node, expanded_node):
        return {}
