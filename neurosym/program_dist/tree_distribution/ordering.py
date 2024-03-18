from abc import ABC, abstractmethod
from types import NoneType
from typing import Dict, List, Union


class NodeOrdering(ABC):
    """
    Represents a technique for ordering the subnodes of a node in a tree.
    """

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
        order = self.compute_order(root_sym_idx)
        if order is None:
            return range(n_subnodes)
        return order


class DictionaryNodeOrdering(NodeOrdering):
    """
    Orders the subnodes of a node according to a dictionary.
    """

    def __init__(self, dist, ordering: Dict[str, List[int]]):
        self.ordering = {dist.symbol_to_index[k]: vs for k, vs in ordering.items()}

    def compute_order(self, root_sym_idx: int) -> Union[List[int], NoneType]:
        return self.ordering.get(root_sym_idx, None)


class DefaultNodeOrdering(NodeOrdering):
    """
    Orders the subnodes of a node according to their symbol indices.
    """

    def compute_order(self, root_sym_idx: int) -> Union[List[int], NoneType]:
        return None
