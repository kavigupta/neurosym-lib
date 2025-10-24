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
    Collects the alernate symbols that could have been selected in the tree distribution,
    and yields the S-Expression, the alternate symbols, and the mask at each node.

    :param s_exp: The S-Expression to collect the alternate symbols for.
    :param tree_dist: The tree distribution.
    :param replace_node_midstream: A function that takes the current node, the mask, the
        position of the node, and the alternate symbols that could have been selected, and
        returns the new node. This is used to replace the node as it is being visited. The
        returned node is then visited in place of the original node.
    :param symbol_to_index_fn: A function that takes the mask and the symbol and returns
        the index of the symbol in the tree distribution.
    """
    mask = tree_dist.mask_constructor(tree_dist)
    mask.on_entry(0, 0)
    yield from _collect_preorder_symbols_dfs(
        s_exp,
        tree_dist,
        mask,
        ((0, 0),),
        replace_node_midstream=replace_node_midstream,
        symbol_to_index_fn=symbol_to_index_fn,
    )


def _collect_preorder_symbols_dfs(
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
        yield from _collect_preorder_symbols_dfs(
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

    Exists mostly for testing purposes.

    :param s_exp: The S-Expression to annotate.
    :param tree_dist: The tree distribution.
    :param summary_fn: A function that takes the chosen symbol and the alternate
        symbols and returns the new symbol.

    :returns: The annotated S-Expression.
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
