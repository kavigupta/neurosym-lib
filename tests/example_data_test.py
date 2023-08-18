"""
NEAR Integration tests.
"""

import unittest
from neurosym.models.mlp import MLP, MLPConfig
from neurosym.models.rnn import Seq2ClassRNN, Seq2SeqRNN, RNNConfig
from neurosym.near.near_graph import near_graph
from neurosym.programs.hole import Hole
from neurosym.programs.s_expression import InitializedSExpression, SExpression

from neurosym.search.bfs import bfs
from neurosym.search.bounded_astar import bounded_astar

from neurosym.examples.example_rnn_dsl import (
    example_rnn_dsl,
)
import torch

from neurosym.search_graph.metadata_computer import NoMetadataComputer
from neurosym.types.type import Type, ArrowType, ListType, TensorType, float_t
from neurosym.data.load_data import numpy_dataset_from_github, DatasetWrapper
from neurosym.types.type_string_repr import TypeDefiner, render_type, parse_type
from neurosym.dsl.neural_dsl import NeuralDSL

import pytest

class TestNEARExample(unittest.TestCase):
    def test_near_astar(self):
        """
        A minimal implementation of NEAR with a simple DSL.
        search = A-star
        heuristic = validation score after training for N epochs. (pl.Trainer)
        goal = Fully symbolic program. (This is handled internally search_graph/dsl_search_graph.py)
        test_predicate = score on testing set (pl.Trainer)
        """
        dataset_gen = numpy_dataset_from_github(
            "https://github.com/trishullab/near/tree/master/near_code/data/example",
            "train_ex_data.npy",
            "train_ex_labels.npy",
            "test_ex_data.npy",
            "test_ex_labels.npy",
        )
        datamodule : DatasetWrapper = dataset_gen(train_seed=0)

        input_shape = datamodule.train.inputs.shape[-1]
        output_shape =  4 # TODO[AS]: remove hardcoded
        t = TypeDefiner(L=input_shape, O=output_shape)
        t.typedef("fL", "{f, $L}")
        t.typedef("fO", "{f, $O}")

        dsl = example_rnn_dsl(10, 4)
        neural_dsl = NeuralDSL.from_dsl(
            dsl=dsl,
            partial_modules= {
                t("($fL) -> $fL") : MLP(MLPConfig(model_name="ll_mlp", input_size=input_shape, hidden_size=10, output_size=input_shape)),
                t("($fL) -> $fO") : MLP(MLPConfig(model_name="lo_mlp", input_size=input_shape, hidden_size=10, output_size=output_shape)),
                t("([$fL]) -> [$fL]") : Seq2SeqRNN(RNNConfig(model_name="ll_rnn", input_size=input_shape, hidden_size=10, output_size=input_shape)),
                t("([$fL]) -> [$fO]") : Seq2SeqRNN(RNNConfig(model_name="lo_rnn", input_size=input_shape, hidden_size=10, output_size=output_shape)),
                t("($fL) -> $fL") : Seq2ClassRNN(RNNConfig(model_name="lc_rnn", input_size=input_shape, hidden_size=10, output_size=input_shape)),
                t("($fL) -> $fO") : Seq2ClassRNN(RNNConfig(model_name="lc_rnn", input_size=input_shape, hidden_size=10, output_size=output_shape)),
            }
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

            neural_dsl.initialize(node.program)
            model = neural_dsl.compute_on_pytorch()
            trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader())
            return trainer.callback_metrics["val_loss"].item()


        g = near_graph(
            neural_dsl,
            parse_type(s="({f, $L}) -> {f, $O}", env=dict(L=input_shape, O=output_shape)),
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

        cost = (
            lambda x: len(str(x.program.children[0]))
            if isinstance(x.program, SExpression) and x.program.children
            else 0
        )

        # succeed if this raises StopIteration
        with pytest.raises(StopIteration):
            node = next(bounded_astar(g, cost, max_depth=7)).program


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

        cost = (
            lambda x: len(str(x.program.children[0]))
            if isinstance(x.program, SExpression) and x.program.children
            else 0
        )
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