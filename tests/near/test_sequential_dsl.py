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

import neurosym as ns
from neurosym.examples import near

from .utils import assertDSLEnumerable


class TestNEARSequentialDSL(unittest.TestCase):
    def test_sequential_dsl_astar(self):
        """
        A minimal implementation of NEAR with a simple DSL.
        search = A-star
        heuristic = validation score after training for N epochs. (pl.Trainer)
        goal = Fully symbolic program. (handled in: search_graph/dsl_search_graph.py)
        test_predicate = score on testing set (pl.Trainer)
        """
        datamodule = ns.datasets.near_data_example(train_seed=0)
        input_dim, output_dim = datamodule.train.get_io_dims()
        original_dsl = near.example_rnn_dsl(input_dim, output_dim)
        trainer_cfg = near.NEARTrainerConfig(
            max_seq_len=100,
            n_epochs=10,
            num_labels=output_dim,
            train_steps=len(datamodule.train),
        )
        t = ns.TypeDefiner(L=input_dim, O=output_dim)
        t.typedef("fL", "{f, $L}")
        t.typedef("fO", "{f, $O}")
        neural_dsl = near.NeuralDSL.from_dsl(
            dsl=original_dsl,
            modules={
                **near.create_modules(
                    "mlp",
                    [t("($fL) -> $fL"), t("($fL) -> $fO")],
                    near.mlp_factory(hidden_size=10),
                ),
                **near.create_modules(
                    "rnn_seq2seq",
                    [t("([$fL]) -> [$fL]"), t("([$fL]) -> [$fO]")],
                    near.rnn_factory_seq2seq(hidden_size=10),
                ),
                **near.create_modules(
                    "rnn_seq2class",
                    [t("([$fL]) -> $fL"), t("([$fL]) -> $fO")],
                    near.rnn_factory_seq2class(hidden_size=10),
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
            except near.PartialProgramNotFoundError:
                return 10000

            model = neural_dsl.compute(initialized_p)
            if not isinstance(model, torch.nn.Module):
                del model
                del initialized_p
                model = near.TorchProgramModule(dsl=neural_dsl, program=node.program)
            pl_model = near.NEARTrainer(model, config=trainer_cfg)
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
            return (
                set(ns.symbols_for_program(node.program)) - set(original_dsl.symbols())
                == set()
            )

        g = near.near_graph(
            neural_dsl,
            ns.parse_type(
                s="([{f, $L}]) -> [{f, $O}]",
                env=ns.TypeDefiner(L=input_dim, O=output_dim),
            ),
            is_goal=checker,
        )
        # succeed if this raises StopIteration
        with pytest.raises(StopIteration):
            n_iter = 0
            iterator = ns.search.bounded_astar(g, validation_cost, max_depth=3)
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
        dsl = near.example_rnn_dsl(10, 4)

        assertDSLEnumerable(dsl, "([$fL]) -> [$fO]")
