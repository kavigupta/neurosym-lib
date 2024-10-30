import uuid
from dataclasses import dataclass, field
from typing import Callable, Generic, Iterable, TypeVar, Union

from neurosym.search_graph.search_graph import SearchGraph
from neurosym.utils.documentation import internal_only

A = TypeVar("A")
B = TypeVar("B")


@dataclass(frozen=True)
@internal_only
class BindSearchGraphNodeA(Generic[A]):
    """
    Represents a node in the initial search graph.

    node: The node in the initial search graph.
    """

    node: A


@dataclass(frozen=True, unsafe_hash=True)
@internal_only
class BindSearchGraphNodeB(Generic[B]):
    """
    Represents a node in the bound search graph (the second one).

    :param node: The node in the second search graph.
    :param graph_b: The second search graph. This is stored here for garbage collection purposes,
    but it could be stored in BindSearchGraph itself. However, that would lead to memory leaks if
    that particular part of the graph is not reachable from the root node. This is why we store it
    here and do not use it for comparison or hashing.
    :param graph_b_uuid: The UUID of the second search graph. This is used to compare nodes in the
    search graph as well as to hash them.
    """

    node: B = field(compare=True, hash=True)
    graph_b: SearchGraph[B] = field(compare=False, hash=False)
    graph_b_uuid: uuid.UUID = field(compare=True, hash=True)

    @internal_only
    def with_node(self, node: B) -> "BindSearchGraphNodeB[B]":
        """
        Returns a new node with the given node.
        """
        return BindSearchGraphNodeB(node, self.graph_b, self.graph_b_uuid)


BindSearchGraphNode = Union[BindSearchGraphNodeA[A], BindSearchGraphNodeB[B]]


class BindSearchGraph(SearchGraph[BindSearchGraphNode[A, B]]):
    """
    Combines two search graphs into one. The first search graph's nodes spawn the second search
    graph.

    This is a monad-like operation that allows us to bind two search graphs together. Unlike
    a monadic bind, the overall graph does not have the type of the second graph's nodes. Instead,
    you need to manually unpack the second graph's nodes.
    """

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
            for neighbor_b in node.graph_b.expand_node(node.node):
                yield node.with_node(neighbor_b)

    def is_goal_node(self, node: BindSearchGraphNode[A, B]) -> bool:
        if isinstance(node, BindSearchGraphNodeA):
            return False
        return node.graph_b.is_goal_node(node.node)

    def cost(self, node: BindSearchGraphNode[A, B]) -> float:
        if isinstance(node, BindSearchGraphNodeA):
            return self.graph_a.cost(node.node)
        return node.graph_b.cost(node.node)
