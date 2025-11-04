import torch

from .near_affine_dsl_builder import NEARAffineSelectorDSLBuilder

BBALL_FEATURES = {
    "ball": torch.LongTensor([0, 1]),
    "offense": torch.LongTensor([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
    "defense": torch.LongTensor([12, 13, 14, 15, 16, 17, 18, 19, 20, 21]),
}
BBALL_FULL_FEATURE_DIM = 22


def simple_bball_dsl(num_classes, hidden_dim=None):
    """
    A differentiable DSL for finding interpretable programs for offensive/defensive behavior
    classification on the BBALL dataset.
    Consult https://arxiv.org/abs/2007.12101 for more details.
    Consult https://github.com/trishullab/near/blob/master/near_code/dsl/basketball.py for the reference implementation.

    :param num_classes: Number of behavior classes.
    :param hidden_dim: Size of hidden dimension (if None, set to num_classes).
    """
    dsl_builder = NEARAffineSelectorDSLBuilder(
        features=BBALL_FEATURES,
        full_feature_dim=BBALL_FULL_FEATURE_DIM,
    )

    return dsl_builder.build_time_invariant(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
    )
