from dataclasses import dataclass, field
from typing import Callable, Generic, Iterable, TypeVar, Union
import uuid

from neurosym.search_graph.search_graph import SearchGraph

A = TypeVar("A")
B = TypeVar("B")


@dataclass(frozen=True)
class BindSearchGraphNodeA(Generic[A]):
    node: A


@dataclass(frozen=True, unsafe_hash=True)
class BindSearchGraphNodeB(Generic[B]):
    node: B = field(compare=True, hash=True)
    # reference is stored here for gc purposes. it could be stored in BindSearchGraph
    # but that would lead to memory leaks if that particular part of the graph is not
    # reachable from the root node.
    graph_b: SearchGraph[B] = field(compare=False, hash=False)
    # uuid is used to identify the graph_b instance. It is used to compare nodes
    # in the search graph as well as to hash them.
    graph_b_uuid: uuid.UUID = field(compare=True, hash=True)

    def with_node(self, node: B) -> "BindSearchGraphNodeB[B]":
        return BindSearchGraphNodeB(node, self.graph_b, self.graph_b_uuid)


BindSearchGraphNode = Union[BindSearchGraphNodeA[A], BindSearchGraphNodeB[B]]


class BindSearchGraph(SearchGraph[BindSearchGraphNode[A, B]]):
    def __init__(
        self, graph_a: SearchGraph[A], create_graph_b: Callable[[A], SearchGraph[B]]
    ):
        self.graph_a = graph_a
        self.create_graph_b = create_graph_b

    def initial_node(self) -> BindSearchGraphNode[A, B]:
        return BindSearchGraphNodeA(self.graph_a.initial_node())

    def expand_node(
        self, node: BindSearchGraphNode[A, B]
    ) -> Iterable[BindSearchGraphNode[A, B]]:
        if isinstance(node, BindSearchGraphNodeA):
            if self.graph_a.is_goal_node(node.node):
                graph_b_uuid = uuid.uuid4()
                graph_b = self.create_graph_b(node.node)
                yield BindSearchGraphNodeB(
                    graph_b.initial_node(), graph_b, graph_b_uuid
                )
            else:
                for neighbor_a in self.graph_a.expand_node(node.node):
                    yield BindSearchGraphNodeA(neighbor_a)
        else:
            graph_b = self.graph_b(node)
            for neighbor_b in graph_b.expand_node(node.node):
                yield node.with_node(neighbor_b)

    def is_goal_node(self, node: BindSearchGraphNode[A, B]) -> bool:
        if isinstance(node, BindSearchGraphNodeA):
            return False
        return node.graph_b.is_goal_node(node.node)

    def cost(self, node: BindSearchGraphNode[A, B]) -> float:
        if isinstance(node, BindSearchGraphNodeA):
            return self.graph_a.cost(node.node)
        return node.graph_b.cost(node.node)
