import numpy as np


def compute_likelihood(tree_dist, program, start_index, start_position, preorder_mask):
    key = start_index + (start_position,)
    syms, log_probs = tree_dist.likelihood_arrays[key]
    mask = preorder_mask.compute_mask(start_position, syms)
    denominator = np.logaddexp.reduce(log_probs[mask])
    top_symbol = tree_dist.symbol_to_index[program.symbol]
    likelihood = (
        tree_dist.distribution_dict[key].get(top_symbol, -float("inf")) - denominator
    )
    if likelihood == -float("inf"):
        return -float("inf")
    preorder_mask.on_entry(start_position, top_symbol)
    for i, child in enumerate(program.children):
        likelihood += compute_likelihood(
            tree_dist,
            child,
            (start_index + (top_symbol,))[-tree_dist.limit :],
            start_position=i,
            preorder_mask=preorder_mask,
        )
    preorder_mask.on_exit(start_position, top_symbol)
    return likelihood
