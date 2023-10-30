import itertools
from abc import abstractmethod

from .search_graph import SearchGraph


class FilterEdgesGraph(SearchGraph):
    def __init__(self, graph):
        self.graph = graph

    @abstractmethod
    def include_edge(self, s, t) -> bool:
        pass

    def initial_node(self):
        return self.graph.initial_node()

    def expand_node(self, node):
        return [t for t in self.graph.expand_node(node) if self.include_edge(node, t)]

    def is_goal_node(self, node):
        return self.graph.is_goal_node(node)


class LimitEdgesGraph(SearchGraph):
    def __init__(self, graph, limit):
        self.graph = graph
        self.limit = limit

    def initial_node(self):
        return self.graph.initial_node()

    def expand_node(self, node):
        return itertools.islice(self.graph.expand_node(node), self.limit)

    def is_goal_node(self, node):
        return self.graph.is_goal_node(node)
