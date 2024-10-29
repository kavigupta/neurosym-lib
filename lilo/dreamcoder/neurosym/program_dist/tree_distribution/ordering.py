from abc import ABC, abstractmethod
from types import NoneType
from typing import Dict, Iterator, List, Union

from neurosym.programs.s_expression import SExpression


class NodeOrdering(ABC):
    """
    Represents a technique for ordering the subnodes of a node in a tree.
    """

    def __init__(self, dist):
        self.dist = dist

    @abstractmethod
    def compute_order(self, root_sym_idx: int) -> Union[List[int], NoneType]:
        """
        Orders the subnodes of the node with the given symbol index.

        None is equivalent to list(range(n_subnodes)).
        """

    def order(self, root_sym_idx: int, n_subnodes: int) -> List[int]:
        """
        Orders the subnodes of the node with the given symbol index.
        """
        assert not isinstance(
            root_sym_idx, str
        ), f"Expected int, got {root_sym_idx} of type {type(root_sym_idx)}"
        order = self.compute_order(root_sym_idx)
        if order is None:
            return range(n_subnodes)
        return order

    def traverse_preorder(self, node: SExpression) -> Iterator[SExpression]:
        """
        Orders the subnodes of the given node.
        """
        yield node
        sym = self.dist.symbol_to_index[node.symbol]
        for i in self.order(sym, len(node.children)):
            yield from self.traverse_preorder(node.children[i])


class DictionaryNodeOrdering(NodeOrdering):
    """
    Orders the subnodes of a node according to a dictionary. The dictionary maps
    symbol indices to lists of integers, which are the indices of the subnodes.

    If a symbol is not in the dictionary, the subnodes are ordered in the default order.

    :param dist: The tree distribution to order.
    :param ordering: The ordering dictionary.
    :param tolerate_missing: Whether to tolerate missing symbols in the ordering. If False,
        an exception will be raised if a symbol in the ordering is not in the distribution.
    """

    def __init__(self, dist, ordering: Dict[str, List[int]], tolerate_missing=False):
        super().__init__(dist)
        if not tolerate_missing:
            assert set(ordering.keys()).issubset(dist.symbol_to_index.keys())
        self.ordering = {
            dist.symbol_to_index[k]: vs
            for k, vs in ordering.items()
            if k in dist.symbol_to_index
        }

    def compute_order(self, root_sym_idx: int) -> Union[List[int], NoneType]:
        return self.ordering.get(root_sym_idx, None)


class DefaultNodeOrdering(NodeOrdering):
    """
    Orders the subnodes of a node in the default order, 0, 1, 2, ...
    """

    def compute_order(self, root_sym_idx: int) -> Union[List[int], NoneType]:
        return None
