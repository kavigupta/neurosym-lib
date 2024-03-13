def compute_likelihood(tree_dist, program, parents):
    top_symbol = tree_dist.symbol_to_index[program.symbol]
    likelihood = tree_dist.distribution_dict[parents].get(top_symbol, -float("inf"))
    if likelihood == -float("inf"):
        return -float("inf")
    for i, child in enumerate(program.children):
        likelihood += compute_likelihood(
            tree_dist,
            child,
            (parents + ((top_symbol, i),))[-tree_dist.limit :],
        )
    return likelihood
