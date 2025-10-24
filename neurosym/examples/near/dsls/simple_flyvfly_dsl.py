# pylint: disable=duplicate-code,cyclic-import
import torch


from .near_affine_dsl_builder import NEARAffineSelectorDSLBuilder


FLYVFLY_FEATURES = {
    "linear": torch.LongTensor([17, 25]),
    "angular": torch.LongTensor([18, 26, 27]),
    "positional": torch.LongTensor([24, 28]),
    "ratio": torch.LongTensor([22, 23]),
    "wing": torch.LongTensor([19, 20, 21]),
    # "FlyStaticFeatures": torch.LongTensor(
    #     [0, 1, 3, 4, 5, 6, 8, 13, 14, 15, 16, 19, 20]
    # ),
    # "FlyDynamicFeatures": torch.LongTensor([17, 18, 22, 23, 24, 27]),
    # "FlyRelativeFeatures": torch.LongTensor(
    #     [25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 45, 46]
    # ),
}
FLYVFLY_FULL_FEATURE_DIM = 53


def simple_flyvfly_dsl(num_classes, seq_len=100, hidden_dim=None):
    """
    A differentiable DSL for finding interpretable programs for fruitfly behavior
    classification on the flyvfly dataset.
    Consult https://arxiv.org/abs/2007.12101 for more details.
    Consult https://github.com/trishullab/near/blob/master/near_code/dsl/fruitflies.py for the reference implementation.

    :param num_classes: Number of behavior classes.
    :param hidden_dim: Size of hidden dimension (if None, set to num_classes).
    """
    dsl_builder = NEARAffineSelectorDSLBuilder(
        features=FLYVFLY_FEATURES,
        full_feature_dim=FLYVFLY_FULL_FEATURE_DIM,
    )

    return dsl_builder.build_time_variant(
        num_classes=num_classes,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
    )
