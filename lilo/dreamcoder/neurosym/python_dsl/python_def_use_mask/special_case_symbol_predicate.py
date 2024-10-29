from abc import ABC, abstractmethod
from typing import List

from neurosym.program_dist.tree_distribution.tree_distribution import TreeDistribution

from .names import GLOBAL_REGEX


class SpecialCaseSymbolPredicate(ABC):
    """
    Handles a special case symbol matching predicate, i.e., a predicate that applies only in
    special cases.

    :param tree_dist: The ``TreeDistribution`` that the predicate is applied to.
    """

    def __init__(self, tree_dist: TreeDistribution):
        self.tree_dist = tree_dist

    @abstractmethod
    def applies(self, symbol: int) -> bool:
        """
        Whether or not the predicate applies to the symbol.

        :param symbol: The symbol to check.
        """

    @abstractmethod
    def compute(self, symbol: int, names: List[str]) -> bool:
        """
        Compute the mask for the given symbol.

        :param symbol: The symbol to compute the mask for.
        :param names: The names available in the current context
        """


class NameEPredicate(SpecialCaseSymbolPredicate):
    """
    Predicate that checks for the ``Name~E`` symbol and allows it only if
    there are global names available or at least one name is present.

    :param tree_dist: The ``TreeDistribution`` that the predicate is applied to.
    """

    def __init__(self, tree_dist: TreeDistribution):
        super().__init__(tree_dist)
        self.has_global_available = any(
            GLOBAL_REGEX.match(x) for x, _ in tree_dist.symbols
        )
        self.name_e = tree_dist.symbol_to_index.get("Name~E", -1)

    def applies(self, symbol: int) -> bool:
        return symbol == self.name_e

    def compute(self, symbol: int, names: List[str]) -> bool:
        return self.has_global_available or len(names) > 0
