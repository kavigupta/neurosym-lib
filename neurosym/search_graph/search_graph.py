from abc import ABC, abstractmethod


class SearchGraph(ABC):
    """
    Where nodes are hashable but can be ids etc.
    """

    @abstractmethod
    def initial_node(self):
        pass

    @abstractmethod
    def expand_node(self, node):
        pass

    @abstractmethod
    def is_goal_node(self, node):
        pass
