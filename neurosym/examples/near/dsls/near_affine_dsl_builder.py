import torch
from torch import nn

from neurosym.dsl.dsl_factory import DSLFactory
from neurosym.types.type import ListType
from neurosym.utils.documentation import internal_only

from ..operations.aggregation import running_agg_torch
from ..operations.basic import ite_torch
from ..operations.lists import map_prefix_torch, map_torch


@internal_only
class NEARAffineSelectorDSLBuilder:
    """
    A builder class for creating differentiable DSLs with affine feature selectors
    for trajectory classification tasks used in the NEAR framework.
    Consult https://arxiv.org/abs/2007.12101 for more details.

    Supports either time-invariant or time-variant DSLs.
    """

    def __init__(self, features: dict[str, torch.LongTensor], full_feature_dim: int):
        """
        Initializes the NEARAffineSelectorDSLBuilder with the given features and full feature dimension.

        :param features: A dictionary mapping feature names to their corresponding indices.
        :param full_feature_dim: The total dimension of the input features.
        """
        self.features = features
        self.full_feature_dim = full_feature_dim

    def build_time_invariant(self, num_classes, hidden_dim=None):
        """
        Build a time-invariant differentiable DSL for trajectory classification.
        The DSL functions are specifically designed for datasets where no advanced aggregations are needed,
        i.e. datasets where we want to predict a class label based on features at each time step independently.

        :param num_classes: Number of behavior classes.
        :param hidden_dim: Size of hidden dimension (if None, set to num_classes).
        :return: A finalized DSL instance.
        """
        hidden_dim = num_classes if hidden_dim is None else hidden_dim

        dslf = DSLFactory(
            input_size=self.full_feature_dim,
            output_size=num_classes,
            max_overall_depth=6,
            hidden_size=hidden_dim,
        )
        dslf.typedef("fO", "{f, $output_size}")
        dslf.typedef("fH", "{f, $hidden_size}")
        dslf.typedef("fI", "{f, $input_size}")

        for feature_name, feature_indices in self.features.items():
            dslf.production(
                f"affine_{feature_name}",
                "() -> $fI -> $fH",
                lambda lin, feature_indices=feature_indices: lambda x: lin(
                    x[..., feature_indices]
                ),
                parameters=dict(
                    lin=lambda feature_indices=feature_indices: nn.Linear(
                        len(feature_indices), hidden_dim
                    )
                ),
            )
            dslf.production(
                f"affine_bool_{feature_name}",
                "() -> $fI -> {f, 1}",
                lambda lin, feature_indices=feature_indices: lambda x: lin(
                    x[..., feature_indices]
                ),
                parameters=dict(
                    lin=lambda feature_indices=feature_indices: nn.Linear(
                        len(feature_indices), 1
                    )
                ),
            )

        dslf.filtered_type_variable(
            "affine_input", lambda x: not isinstance(x, ListType)
        )

        dslf.production(
            "add",
            "(%affine_input -> #b, %affine_input -> #b) -> %affine_input -> #b",
            lambda f1, f2: lambda x: f1(x) + f2(x),
        )
        dslf.production(
            "mul",
            "(%affine_input -> #b, %affine_input -> #b) -> %affine_input -> #b",
            lambda f1, f2: lambda x: f1(x) * f2(x),
        )

        dslf.production(
            "running_avg_last5",
            "(#a -> $fH) -> [#a] -> $fH",
            lambda f: lambda x: running_agg_torch(x, f, lambda t: t - 4, lambda t: t),
        )
        dslf.production(
            "running_avg_last10",
            "(#a -> $fH) -> [#a] -> $fH",
            lambda f: lambda x: running_agg_torch(x, f, lambda t: t - 9, lambda t: t),
        )

        dslf.production(
            "running_avg_window5",
            "(#a -> $fH) -> [#a] -> $fH",
            lambda f: lambda x: running_agg_torch(
                x, f, lambda t: t - 2, lambda t: t + 2
            ),
        )
        dslf.production(
            "running_avg_window11",
            "(#a -> $fH) -> [#a] -> $fH",
            lambda f: lambda x: running_agg_torch(
                x, f, lambda t: t - 5, lambda t: t + 5
            ),
        )

        if hidden_dim != num_classes:
            dslf.production(
                "output",
                "(([$fI]) -> [$fH]) -> [$fI] -> [$fO]",
                lambda f, lin: lambda x: lin(f(x)).softmax(-1),
                dict(lin=lambda: nn.Linear(hidden_dim, num_classes)),
            )
        dslf.production(
            "ite",
            "(#a -> {f, 1}, #a -> #b, #a -> #b) -> #a -> #b",
            ite_torch,
        )
        dslf.production(
            "map",
            "(#a -> #b) -> [#a] -> [#b]",
            lambda f: lambda x: map_torch(f, x),
        )
        dslf.production(
            "map_prefix",
            "([#a] -> #b) -> [#a] -> [#b]",
            lambda f: lambda x: map_prefix_torch(f, x),
        )

        dslf.prune_to("([$fI]) -> [$fO]")
        return dslf.finalize()

    def build_time_variant(self, num_classes, seq_len, hidden_dim=None):
        """
        Build a time-variant differentiable DSL for trajectory classification.
        The DSL functions are specifically designed for datasets where we want to predict a class label
        based on features across the entire trajectory, allowing for temporal aggregations.

        i.e., given an input sequence of shape (seq_len, feature_dim), we want to predict a class label
        (num_classes) based on the entire sequence.

        :param num_classes: Number of behavior classes.
        :param seq_len: Length of the input sequences.
        :param hidden_dim: Size of hidden dimension (if None, set to num_classes).
        :return: A finalized DSL instance.
        """
        hidden_dim = num_classes if hidden_dim is None else hidden_dim
        dslf = DSLFactory(
            input_size=self.full_feature_dim,
            output_size=num_classes,
            max_overall_depth=6,
            hidden_size=hidden_dim,
        )
        dslf.typedef("fO", "{f, $output_size}")
        dslf.typedef("fH", "{f, $hidden_size}")
        dslf.typedef("fI", "{f, $input_size}")

        for feature_name, feature_indices in self.features.items():
            dslf.production(
                f"affine_{feature_name}",
                "() -> $fI -> $fH",
                lambda lin, feature_indices=feature_indices: lambda x: lin(
                    x[..., feature_indices]
                ),
                parameters=dict(
                    lin=lambda feature_indices=feature_indices: nn.Linear(
                        len(feature_indices), hidden_dim
                    )
                ),
            )
            dslf.production(
                f"affine_bool_{feature_name}",
                "() -> $fI -> {f, 1}",
                lambda lin, feature_indices=feature_indices: lambda x: lin(
                    x[..., feature_indices]
                ),
                parameters=dict(
                    lin=lambda feature_indices=feature_indices: nn.Linear(
                        len(feature_indices), 1
                    )
                ),
            )

        dslf.filtered_type_variable(
            "affine_input", lambda x: not isinstance(x, ListType)
        )

        dslf.production(
            "add",
            r"(%affine_input -> #b, %affine_input -> #b) -> %affine_input -> #b",
            lambda f1, f2: lambda x: f1(x) + f2(x),
        )
        dslf.production(
            "mul",
            r"(%affine_input -> #b, %affine_input -> #b) -> %affine_input -> #b",
            lambda f1, f2: lambda x: f1(x) * f2(x),
        )

        def _aggregate_over_windows(seq_fn, x, window_start, window_end):
            seq = seq_fn(x)
            return running_agg_torch(
                seq, lambda y: y, window_start, window_end
            )

        dslf.production(
            "running_avg_last5",
            "([#a] -> [$fH]) -> [#a] -> $fH",
            lambda f: lambda x: _aggregate_over_windows(
                f, x, lambda t: t - 4, lambda t: t
            ),
        )
        dslf.production(
            "running_avg_last10",
            "([#a] -> [$fH]) -> [#a] -> $fH",
            lambda f: lambda x: _aggregate_over_windows(
                f, x, lambda t: t - 9, lambda t: t
            ),
        )

        dslf.production(
            "running_avg_window5",
            "([#a] -> [$fH]) -> [#a] -> $fH",
            lambda f: lambda x: _aggregate_over_windows(
                f, x, lambda t: t - 2, lambda t: t + 2
            ),
        )
        dslf.production(
            "running_avg_window11",
            "([#a] -> [$fH]) -> [#a] -> $fH",
            lambda f: lambda x: _aggregate_over_windows(
                f, x, lambda t: t - 5, lambda t: t + 5
            ),
        )

        dslf.production(
            f"convolve_3_len{seq_len}",
            "([#a] -> [$fH]) -> [#a] -> $fH",
            lambda f, conv: lambda x: conv(f(x)).squeeze(),
            parameters=dict(
                conv=lambda: torch.nn.Conv1d(seq_len, 1, 3, padding=1, bias=False)
            ),
        )
        dslf.production(
            f"convolve_5_len{seq_len}",
            "([#a] -> [$fH]) -> [#a] -> $fH",
            lambda f, conv: lambda x: conv(f(x)).squeeze(),
            parameters=dict(
                conv=lambda: torch.nn.Conv1d(seq_len, 1, 5, padding=2, bias=False)
            ),
        )

        if hidden_dim != num_classes:
            dslf.production(
                "output",
                "(([$fI]) -> $fH) -> [$fI] -> $fO",
                lambda f, lin: lambda x: lin(f(x)).softmax(-1),
                dict(lin=lambda: nn.Linear(hidden_dim, num_classes)),
            )

        dslf.production(
            "ite",
            "(#a -> {f, 1}, #a -> #b, #a -> #b) -> #a -> #b",
            ite_torch,
        )
        # pylint: enable=unnecessary-lambda
        dslf.production(
            "map",
            "(#a -> #b) -> [#a] -> [#b]",
            lambda f: lambda x: map_torch(f, x),
        )
        dslf.production(
            "map_prefix",
            "([#a] -> #b) -> [#a] -> [#b]",
            lambda f: lambda x: map_prefix_torch(f, x),
        )

        dslf.prune_to("[$fI] -> $fO")
        return dslf.finalize()
