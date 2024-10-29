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

        g = near.near_graph(
            neural_dsl,
            ns.parse_type(
                s="([{f, $L}]) -> [{f, $O}]",
                env=ns.TypeDefiner(L=input_dim, O=output_dim),
            ),
            is_goal=neural_dsl.program_has_no_holes,
        )
        # succeed if this raises StopIteration
        with pytest.raises(StopIteration):
            n_iter = 0
            iterator = ns.search.bounded_astar(
                g,
                near.ValidationCost(
                    neural_dsl=neural_dsl,
                    trainer_cfg=trainer_cfg,
                    datamodule=datamodule,
                ),
                max_depth=3,
            )
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
