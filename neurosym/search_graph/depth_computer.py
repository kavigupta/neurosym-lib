from abc import ABC, abstractmethod



class DepthComputer(ABC):
    """
    Represents a way to compute depth for a node in a search graph.
    """

    @abstractmethod
    def initialize(self):
        """
        Return the Depth for the initial node in the search graph.
        """

    @abstractmethod
    def probable_depth(self, node, current_depth):
        """
        Return the Depth for the expanded node in the search graph.
        """

    @abstractmethod
    def increment(self, node, current_depth):
        """
        Return the Depth for the expanded node in the search graph.
        """


class UniformDepthComputer(DepthComputer):
    """
    Represents a Depth computer that increments depth by 1 for all program structures.
    """

    def initialize(self):
        return 0

    def probable_depth(self, node, current_depth):
        return current_depth

    def increment(self, node, current_depth):
        return current_depth + 1
