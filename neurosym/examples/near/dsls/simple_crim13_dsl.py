import torch

from .near_affine_dsl_builder import NEARAffineSelectorDSLBuilder

CRIM13_FEATURES = {
    "position": torch.LongTensor([0, 1, 2, 3]),
    "distance": torch.LongTensor([4]),
    "distance_change": torch.LongTensor([5]),
    "angle": torch.LongTensor([6, 7, 10]),
    "angle_change": torch.LongTensor([8, 9]),
    "velocity": torch.LongTensor([11, 12, 13, 14]),
    "acceleration": torch.LongTensor([15, 16, 17, 18]),
}

CRIM13_FULL_FEATURE_DIM = 19


def simple_crim13_dsl(num_classes, hidden_dim=None):
    """
    A differentiable DSL for finding interpretable programs for mice behavior
    classification on the CRIM13 dataset.
    Consult https://arxiv.org/abs/2007.12101 for more details.
    Consult https://github.com/trishullab/near/blob/master/near_code/dsl_crim13.py for the reference implementation.

    :param num_classes: Number of behavior classes.
    :param hidden_dim: Size of hidden dimension (if None, set to num_classes).
    """
    dsl_builder = NEARAffineSelectorDSLBuilder(
        features=CRIM13_FEATURES,
        full_feature_dim=CRIM13_FULL_FEATURE_DIM,
    )

    return dsl_builder.build_time_invariant(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
    )
