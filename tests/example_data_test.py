"""
NEAR Integration tests.
"""

import unittest
from neurosym.models.mlp import MLP, MLPConfig
from neurosym.models.rnn import Seq2ClassRNN, Seq2SeqRNN, RNNConfig
from neurosym.near.near_graph import near_graph
from neurosym.programs.s_expression import SExpression

from neurosym.search.bounded_astar import bounded_astar

from neurosym.examples.example_rnn_dsl import (
    example_rnn_dsl,
)
import torch

from neurosym.types.type import ArrowType, ListType, TensorType, float_t
from neurosym.data.load_data import numpy_dataset_from_github, DatasetWrapper
from neurosym.types.type_string_repr import TypeDefiner, parse_type
from neurosym.dsl.neural_dsl import NeuralDSL, create_modules

import pytest


class TestNEARExample(unittest.TestCase):
    def test_near_astar(self):
        """
        A minimal implementation of NEAR with a simple DSL.
        search = A-star
        heuristic = validation score after training for N epochs. (pl.Trainer)
        goal = Fully symbolic program. (handled in: search_graph/dsl_search_graph.py)
        test_predicate = score on testing set (pl.Trainer)
        """
        dataset_gen = numpy_dataset_from_github(
            "https://github.com/trishullab/near/tree/master/near_code/data/example",
            "train_ex_data.npy",
            "train_ex_labels.npy",
            "test_ex_data.npy",
            "test_ex_labels.npy",
        )
        datamodule: DatasetWrapper = dataset_gen(train_seed=0)

        input_shape = datamodule.train.inputs.shape[-1]
        output_shape = 4  # TODO[AS]: remove hardcoded
        t = TypeDefiner(L=input_shape, O=output_shape)
        t.typedef("fL", "{f, $L}")
        t.typedef("fO", "{f, $O}")

        dsl = example_rnn_dsl(10, 4)

        # TODO [AS]: move to a separate file

        def mlp_factory(**kwargs):
            return lambda input_shape, output_shape: MLP(
                MLPConfig(
                    model_name="mlp",
                    input_size=input_shape,
                    output_size=output_shape,
                    **kwargs,
                )
            )

        def rnn_factory(**kwargs):
            return lambda input_shape, output_shape: Seq2SeqRNN(
                RNNConfig(
                    model_name="rnn",
                    input_size=input_shape,
                    output_size=output_shape,
                    **kwargs,
                )
            )

        def rnn_factory_seq2class(**kwargs):
            return lambda input_shape, output_shape: Seq2ClassRNN(
                RNNConfig(
                    model_name="rnn",
                    input_size=input_shape,
                    output_size=output_shape,
                    **kwargs,
                )
            )

        neural_dsl = NeuralDSL.from_dsl(
            dsl=dsl,
            modules={
                **create_modules(
                    [t("($fL) -> $fL"), t("($fL) -> $fO")],
                    mlp_factory(hidden_size=10),
                ),
                **create_modules(
                    [t("([$fL]) -> [$fL]"), t("([$fL]) -> [$fO]")],
                    rnn_factory(hidden_size=10),
                ),
                **create_modules(
                    [t("([$fL]) -> f"), t("([$fL]) -> $fO")],
                    rnn_factory_seq2class(hidden_size=10),
                ),
            },
        )

        def validation_cost(node):
            import pytorch_lightning as pl

            trainer = pl.Trainer(
                max_epochs=10,
                # devices=0,
                # accelerator="auto",
                logger=False,
                callbacks=[],
            )

            initialized_p = neural_dsl.initialize(node.program)
            model = neural_dsl.compute_on_pytorch(initialized_p)
            trainer.fit(
                model, datamodule.train_dataloader(), datamodule.val_dataloader()
            )
            return trainer.callback_metrics["val_loss"].item()

        g = near_graph(
            neural_dsl,
            parse_type(
                s="({f, $L}) -> {f, $O}", env=dict(L=input_shape, O=output_shape)
            ),
            is_goal=lambda node: True,
        )
        node = next(bounded_astar(g, validation_cost, max_depth=7)).program
        print(node)

    def test_dsl(self):
        """
        Enumerate all programs in example_rnn_dsl upto
        fixed depth.
        This test case just makes sure all DSL combinations
        upto a fixed depth are valid.
        """
        self.maxDiff = None
        input_size = 10
        output_size = 4
        dsl = example_rnn_dsl(input_size, 4)

        def checker(x):
            """Initialize and return False"""
            x = x.program
            xx = dsl.compute_on_pytorch(dsl.initialize(x))
            print(xx)
            return False

        g = near_graph(
            dsl,
            ArrowType(
                (ListType(TensorType(float_t, (input_size,))),),
                ListType(TensorType(float_t, (output_size,))),
            ),
            is_goal=checker,
        )

        def cost(x):
            if isinstance(x.program, SExpression) and x.program.children:
                return len(str(x.program.children[0]))
            return 0

        # succeed if this raises StopIteration
        with pytest.raises(StopIteration):
            next(bounded_astar(g, cost, max_depth=7)).program

    def synthetic_test_near_astar(self):
        self.maxDiff = None
        input_size = 10
        dsl = example_rnn_dsl(input_size, 4)
        fours = torch.full((input_size,), 4.0)

        # in folder examples/example_rnn, run
        # train_ex_data.npy  train_ex_labels.npy

        def checker(x):
            x = x.program
            xx = dsl.compute_on_pytorch(dsl.initialize(x))
            if isinstance(xx, torch.Tensor):
                return torch.all(torch.eq(xx, fours))
            else:
                return False

        g = near_graph(
            dsl,
            ArrowType(
                (ListType(TensorType(float_t, (input_size,))),),
                ListType(TensorType(float_t, (4,))),
            ),
            is_goal=checker,
        )

        def cost(x):
            if isinstance(x.program, SExpression) and x.program.children:
                return len(str(x.program.children[0]))
            return 0

        node = next(bounded_astar(g, cost, max_depth=7)).program
        self.assertEqual(
            node,
            SExpression(
                symbol="Tint_int_add",
                children=(
                    SExpression(symbol="ones", children=()),
                    SExpression(
                        symbol="int_int_add",
                        children=(
                            SExpression(
                                symbol="int_int_add",
                                children=(
                                    SExpression(symbol="one", children=()),
                                    SExpression(symbol="one", children=()),
                                ),
                            ),
                            SExpression(symbol="one", children=()),
                        ),
                    ),
                ),
            ),
        )


if __name__ == "__main__":
    print("Running tests/example_data_test.py")
    TestNEARExample().test_near_astar()
