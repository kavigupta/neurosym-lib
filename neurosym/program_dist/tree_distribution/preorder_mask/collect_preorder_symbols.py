from typing import Callable, Iterator, List, Tuple

import numpy as np

from neurosym.program_dist.tree_distribution.preorder_mask.preorder_mask import (
    PreorderMask,
)
from neurosym.program_dist.tree_distribution.tree_distribution import TreeDistribution
from neurosym.programs.s_expression import SExpression


def collect_preorder_symbols(
    s_exp: SExpression,
    tree_dist: TreeDistribution,
    mask: Callable[[TreeDistribution], PreorderMask],
) -> Iterator[Tuple[SExpression, List[str]]]:
    """
    Collects the alernate symbols that could have been selected in the tree distribution.
    """
    mask = mask(tree_dist)
    mask.on_entry(0, 0)
    yield from collect_preorder_symbols_dfs(s_exp, tree_dist, mask, 0)


def collect_preorder_symbols_dfs(
    s_exp: SExpression,
    tree_dist: TreeDistribution,
    mask: PreorderMask,
    position: int,
) -> Iterator[Tuple[SExpression, List[str]]]:
    """
    Collects the alernate symbols that could have been selected in the tree distribution.
    """
    print(s_exp, position)
    print(mask.type_stack)
    idxs = np.arange(len(tree_dist.symbols))
    yield s_exp, tuple(int(x) for x in idxs[mask.compute_mask(position, idxs)])
    mask.on_entry(position, tree_dist.symbol_to_index[s_exp.symbol])
    for idx, child in enumerate(s_exp.children):
        yield from collect_preorder_symbols_dfs(child, tree_dist, mask, idx)
    mask.on_exit(position, tree_dist.symbol_to_index[s_exp.symbol])


def annotate_with_alternate_symbols(
    s_exp: SExpression,
    tree_dist: TreeDistribution,
    mask: Callable[[TreeDistribution], PreorderMask],
    summary_fn=lambda chosen, alts: f"{chosen}/{','.join(sorted(alts))}",
) -> SExpression:
    """
    Annotates the S-Expression with the alternate symbols that could have been selected in the tree distribution.
    """
    node_id_to_alts = {
        id(node): tuple(tree_dist.symbols[alt][0] for alt in alts)
        for node, alts in collect_preorder_symbols(s_exp, tree_dist, mask)
    }

    def replace(s):
        symbol = summary_fn(s.symbol, node_id_to_alts[id(s)])
        children = [replace(c) for c in s.children]
        return SExpression(symbol, children)

    return replace(s_exp)
