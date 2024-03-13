import numpy as np


def compute_likelihood(tree_dist, program, parents, preorder_mask):
    syms, log_probs = tree_dist.likelihood_arrays[parents]
    start_position = parents[-1][1]
    mask = preorder_mask.compute_mask(start_position, syms)
    denominator = np.logaddexp.reduce(log_probs[mask])
    top_symbol = tree_dist.symbol_to_index[program.symbol]
    likelihood = (
        tree_dist.distribution_dict[parents].get(top_symbol, -float("inf"))
        - denominator
    )
    if likelihood == -float("inf"):
        return -float("inf")
    preorder_mask.on_entry(start_position, top_symbol)
    for i, child in enumerate(program.children):
        likelihood += compute_likelihood(
            tree_dist,
            child,
            (parents + ((top_symbol, i),))[-tree_dist.limit :],
            preorder_mask=preorder_mask,
        )
    preorder_mask.on_exit(start_position, top_symbol)
    return likelihood
