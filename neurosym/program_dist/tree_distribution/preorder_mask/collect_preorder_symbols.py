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
    replace_node_midstream: Callable[
        [SExpression, PreorderMask, int, List[int]], SExpression
    ] = None,
    symbol_to_index_fn=None,
) -> Iterator[Tuple[SExpression, List[str], PreorderMask]]:
    """
    Collects the alernate symbols that could have been selected in the tree distribution.
    """
    mask = tree_dist.mask_constructor(tree_dist)
    mask.on_entry(0, 0)
    yield from collect_preorder_symbols_dfs(
        s_exp,
        tree_dist,
        mask,
        ((0, 0),),
        replace_node_midstream=replace_node_midstream,
        symbol_to_index_fn=symbol_to_index_fn,
    )


def collect_preorder_symbols_dfs(
    s_exp: SExpression,
    tree_dist: TreeDistribution,
    mask: PreorderMask,
    parents: Tuple[Tuple[int, int], ...],
    replace_node_midstream: Callable[
        [SExpression, PreorderMask, int, List[int]], SExpression
    ] = None,
    symbol_to_index_fn=None,
) -> Iterator[Tuple[SExpression, List[str], PreorderMask]]:
    """
    Collects the alernate symbols that could have been selected in the tree distribution.
    """
    position = parents[-1][1]
    idxs = np.array([i for i, _ in tree_dist.distribution[parents]])
    bool_mask = mask.compute_mask(position, idxs)
    alts = tuple(int(x) for x in idxs[bool_mask])
    if replace_node_midstream is not None:
        s_exp = replace_node_midstream(s_exp, mask, position, alts)
    yield s_exp, alts, mask
    sym_idx = (
        tree_dist.symbol_to_index[s_exp.symbol]
        if symbol_to_index_fn is None
        else symbol_to_index_fn(mask, s_exp.symbol)
    )
    mask.on_entry(position, sym_idx)
    order = tree_dist.ordering.order(sym_idx, len(s_exp.children))
    for idx, child in zip(order, [s_exp.children[i] for i in order]):
        new_parents = (parents + ((sym_idx, idx),))[-tree_dist.limit :]
        yield from collect_preorder_symbols_dfs(
            child,
            tree_dist,
            mask,
            new_parents,
            replace_node_midstream=replace_node_midstream,
            symbol_to_index_fn=symbol_to_index_fn,
        )
    mask.on_exit(position, sym_idx)


def annotate_with_alternate_symbols(
    s_exp: SExpression,
    tree_dist: TreeDistribution,
    summary_fn=lambda chosen, alts: f"{chosen}/{','.join(sorted(alts))}",
) -> SExpression:
    """
    Annotates the S-Expression with the alternate symbols that could have been
    selected in the tree distribution.
    """
    preorder_symbols = [
        (node, alts) for node, alts, _ in collect_preorder_symbols(s_exp, tree_dist)
    ]
    assert len(preorder_symbols) == len({id(node) for node, _ in preorder_symbols})
    node_id_to_alts = {
        id(node): tuple(tree_dist.symbols[alt][0] for alt in alts)
        for node, alts in preorder_symbols
    }

    def replace(s):
        symbol = summary_fn(s.symbol, node_id_to_alts[id(s)])
        children = [replace(c) for c in s.children]
        return SExpression(symbol, children)

    return replace(s_exp)
