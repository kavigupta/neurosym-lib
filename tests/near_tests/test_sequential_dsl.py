"""
Test dsl/sequentual_dsl.py with NEAR search graph.

We conduct the following tests:
- Sanity check: We can find a program.
- BFS: We can find a program with BFS.
- Astar: We can find a program with bounded Astar search.
- Enumerate: We can enumerate all programs of a certain size.
- Full Integration test: Can we run a full iteration of NEAR.
NEAR Integration tests.
"""

import unittest

import pytest
import torch

from neurosym.near.datasets.load_data import DatasetWrapper, numpy_dataset_from_github
from neurosym.near.dsls.sequential_differentiable_dsl import example_rnn_dsl
from neurosym.near.methods.near_example_trainer import NEARTrainer, NEARTrainerConfig
from neurosym.near.models.mlp import mlp_factory
from neurosym.near.models.rnn import rnn_factory_seq2class, rnn_factory_seq2seq
from neurosym.near.models.torch_program_module import TorchProgramModule
from neurosym.near.neural_dsl import (
    NeuralDSL,
    PartialProgramNotFoundError,
    create_modules,
)
from neurosym.near.search_graph import near_graph
from neurosym.programs.s_expression import SExpression
from neurosym.programs.s_expression_render import symbols
from neurosym.search.bounded_astar import bounded_astar
from neurosym.types.type_string_repr import TypeDefiner, parse_type


class TestNEARSequentialDSL(unittest.TestCase):
    def test_sequential_dsl_astar(self):
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
        input_dim, output_dim = datamodule.train.get_io_dims()
        original_dsl = example_rnn_dsl(input_dim, output_dim)
        trainer_cfg = NEARTrainerConfig(
            max_seq_len=100,
            n_epochs=10,
            num_labels=output_dim,
            train_steps=len(datamodule.train),
        )
        t = TypeDefiner(L=input_dim, O=output_dim)
        t.typedef("fL", "{f, $L}")
        t.typedef("fO", "{f, $O}")
        neural_dsl = NeuralDSL.from_dsl(
            dsl=original_dsl,
            modules={
                **create_modules(
                    "mlp",
                    [t("($fL) -> $fL"), t("($fL) -> $fO")],
                    mlp_factory(hidden_size=10),
                ),
                **create_modules(
                    "rnn_seq2seq",
                    [t("([$fL]) -> [$fL]"), t("([$fL]) -> [$fO]")],
                    rnn_factory_seq2seq(hidden_size=10),
                ),
                **create_modules(
                    "rnn_seq2class",
                    [t("([$fL]) -> $fL"), t("([$fL]) -> $fO")],
                    rnn_factory_seq2class(hidden_size=10),
                ),
            },
        )

        def validation_cost(node):
            import pytorch_lightning as pl

            trainer = pl.Trainer(
                max_epochs=10,
                devices="auto",
                accelerator="cpu",
                enable_checkpointing=False,
                logger=False,
                callbacks=[],
            )
            try:
                initialized_p = neural_dsl.initialize(node.program)
            except PartialProgramNotFoundError:
                return 10000

            model = neural_dsl.compute(initialized_p)
            if not isinstance(model, torch.nn.Module):
                del model
                del initialized_p
                model = TorchProgramModule(dsl=neural_dsl, program=node.program)
            pl_model = NEARTrainer(model, config=trainer_cfg)
            trainer.fit(
                pl_model, datamodule.train_dataloader(), datamodule.val_dataloader()
            )
            return trainer.callback_metrics["val_loss"].item()

        def checker(node):
            """
            In NEAR, any program that has no holes is valid.
            The hole checking is done before this function will
            be called so we can assume that the program has no holes.
            """
            return set(symbols(node.program)) - set(original_dsl.symbols()) == set()

        g = near_graph(
            neural_dsl,
            parse_type(
                s="([{f, $L}]) -> [{f, $O}]", env=dict(L=input_dim, O=output_dim)
            ),
            is_goal=checker,
        )
        # succeed if this raises StopIteration
        with pytest.raises(StopIteration):
            n_iter = 0
            iterator = bounded_astar(g, validation_cost, max_depth=3)
            while True:
                print("iteration: ", n_iter)
                n_iter += 1
                node = next(iterator)
                self.assertIsNotNone(node)
                if n_iter > 30:
                    break

    def test_sequential_dsl_enumerate(self):
        """
        Enumerate all programs in dsl upto fixed depth. This test case makes
        sure all DSL combinations upto a fixed depth are valid.
        """
        self.maxDiff = None
        input_dim = 10
        output_dim = 4
        max_depth = 5
        t = TypeDefiner(L=input_dim, O=output_dim)
        t.typedef("fL", "{f, $L}")
        t.typedef("fO", "{f, $O}")
        dsl = example_rnn_dsl(input_dim, output_dim)

        def checker(x):
            """Initialize and return True"""
            x = x.program
            xx = dsl.compute(dsl.initialize(x))
            print(xx)
            return True

        g = near_graph(
            dsl,
            t("([$fL]) -> [$fO]"),
            is_goal=checker,
        )

        def cost(x):
            if isinstance(x.program, SExpression) and x.program.children:
                return len(str(x.program.children[0]))
            return 0

        # succeed if this raises StopIteration
        for _ in bounded_astar(g, cost, max_depth=max_depth):
            pass
