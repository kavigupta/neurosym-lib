from abc import ABC, abstractmethod
from typing import List

import numpy as np


class PreorderMask(ABC):
    """
    Represents a mask on symbols that is updated during a preorder traversal of the tree.

    This can be used to implement such features as type checking and other constraints
        during enumeration, sampling, likelihood computation, and counting.
    """

    def __init__(self, tree_dist):
        self.tree_dist = tree_dist

    @abstractmethod
    def compute_mask(self, position: int, symbols: List[int]) -> List[bool]:
        """
        Compute the mask at the current position with the current symbols.

        The mask should be True for symbols that are allowed and False for symbols that are not.
        """

    @abstractmethod
    def on_entry(self, position: int, symbol: int):
        """
        Called when entering a node in the preorder traversal.

        This can be used to update the mask.
        """

    @abstractmethod
    def on_exit(self, position: int, symbol: int):
        """
        Called when exiting a node in the preorder traversal.

        This can be used to update the mask.
        """


class NoopPreorderMask(PreorderMask):
    """
    A mask that does nothing.
    """

    def compute_mask(self, position: int, symbols: List[int]) -> List[bool]:
        return [True] * len(symbols)

    def on_entry(self, position: int, symbol: int):
        pass

    def on_exit(self, position: int, symbol: int):
        pass


class ConjunctionPreorderMask(PreorderMask):
    """
    A mask that is the conjunction of multiple masks.
    """

    def __init__(self, tree_dist, masks):
        super().__init__(tree_dist)
        self.masks = masks

    def compute_mask(self, position: int, symbols: List[int]) -> List[bool]:
        symbols = np.array(symbols)
        mask = np.ones(len(symbols), dtype=bool)
        for m in self.masks:
            [valid_symbol_idxs] = np.where(mask)
            mask[valid_symbol_idxs] &= m.compute_mask(
                position, symbols[valid_symbol_idxs]
            )
        return mask.tolist()

    def on_entry(self, position: int, symbol: int):
        for mask in self.masks:
            mask.on_entry(position, symbol)

    def on_exit(self, position: int, symbol: int):
        for mask in self.masks:
            mask.on_exit(position, symbol)
