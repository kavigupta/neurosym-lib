def compute_likelihood(tree_dist, program, start_index, start_position):
    key = start_index + (start_position,)
    top_symbol = tree_dist.symbol_to_index[program.symbol]
    likelihood = tree_dist.distribution_dict[key].get(top_symbol, -float("inf"))
    if likelihood == -float("inf"):
        return -float("inf")
    for i, child in enumerate(program.children):
        likelihood += compute_likelihood(
            tree_dist,
            child,
            (start_index + (top_symbol,))[-tree_dist.limit :],
            start_position=i,
        )
    return likelihood
