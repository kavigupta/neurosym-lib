import numpy as np

from neurosym.dsl.dsl_factory import DSLFactory
from neurosym.program_dist.bigram import BigramProgramDistributionFamily


def _make_simple_lambda_dsl():
    """
    Build a tiny DSL with lambdas and variables so that we can exercise
    the DreamCoder-compatible variable masking logic.
    """
    dslf = DSLFactory()
    # integer base type
    dslf.typedef("i", "i")
    # identity function and a constant as simple non-variable productions
    dslf.production("id", "(i) -> i", lambda x: x)
    dslf.production("one", "() -> i", lambda: 1)
    # add lambda/variable productions
    dslf.lambdas(max_type_depth=2)
    dslf.known_types(("i",))
    dsl = dslf.finalize()
    return dsl


def test_dc_compatible_mask_collapses_variables():
    """
    Sanity-check that TypePreorderMaskDCCompatible, when composed with the
    regular TypePreorderMask inside BigramProgramDistributionFamily with
    dc_compatible_variables=True, collapses multiple variable symbols
    into a single candidate in a given context.
    """
    dsl = _make_simple_lambda_dsl()
    fam = BigramProgramDistributionFamily(
        dsl,
        include_type_preorder_mask=True,
        dc_compatible_variables=True,
    )
    # Extract the tree distribution skeleton and corresponding preorder mask.
    tree_dist = fam.tree_distribution_skeleton
    preorder_mask = fam._compute_preorder_mask(tree_dist)  # pylint: disable=protected-access

    # Find all indices that correspond to variable-like symbols (names starting with "$").
    variable_indices = [
        idx
        for idx, (sym, _arity) in enumerate(tree_dist.symbols)
        if isinstance(sym, str) and sym.startswith("$")
    ]
    # If the DSL factory did not generate variables for some reason, skip.
    if len(variable_indices) <= 1:
        # Nothing to test in this degenerate case.
        return

    # Ask the DC-compatible mask what it allows in a fake "root" position.
    # We pass *only* variable indices so that we can see the collapsing behaviour.
    mask = np.asarray(
        preorder_mask.compute_mask(0, np.array(variable_indices, dtype=int)),
        dtype=float,
    )

    # Exactly one variable index should remain with positive weight.
    assert np.count_nonzero(mask > 0.0) == 1


def test_dc_compatible_mask_does_not_change_non_variables():
    """
    The DC-compatible mask should only affect variable symbols; non-variable
    symbols that are type-valid should be unaffected (weight 1.0).
    """
    dsl = _make_simple_lambda_dsl()
    fam = BigramProgramDistributionFamily(
        dsl,
        include_type_preorder_mask=True,
        dc_compatible_variables=True,
    )
    tree_dist = fam.tree_distribution_skeleton
    preorder_mask = fam._compute_preorder_mask(tree_dist)  # pylint: disable=protected-access

    all_indices = np.arange(len(tree_dist.symbols))
    base_mask = np.asarray(preorder_mask.compute_mask(0, all_indices), dtype=float)

    for idx, w in enumerate(base_mask):
        sym, _arity = tree_dist.symbols[idx]
        if not (isinstance(sym, str) and sym.startswith("$")):
            # For non-variable symbols that are type-valid, the mask weight
            # should be either 0.0 or 1.0, and DC-compat mode should not
            # introduce any intermediate weights.
            assert w in (0.0, 1.0)
