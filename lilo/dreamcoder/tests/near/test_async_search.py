"""
Test search/bounded_astar_async.py with NEAR search graph.
"""

import unittest

import pytest

import neurosym as ns
from neurosym.examples import near


class TestNEARAsyncSearch(unittest.TestCase):
    def test_astar_async_sequential_dsl(self):
        """
        A minimal implementation of NEAR with a simple DSL.
        search = A-star
        heuristic = validation score after training for N epochs. (pl.Trainer)
        goal = Fully symbolic program. (handled in: search_graph/dsl_search_graph.py)
        test_predicate = score on testing set (pl.Trainer)

        This tests an async version of bounded_astar search.
        """
        # pylint: disable=duplicate-code
        datamodule = ns.datasets.near_data_example(
            train_seed=0, batch_size=32, num_workers=0
        )
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
        max_depth = 3

        g = near.near_graph(
            neural_dsl,
            ns.parse_type(
                s="([{f, $L}]) -> [{f, $O}]",
                env=ns.TypeDefiner(L=input_dim, O=output_dim),
            ),
            is_goal=neural_dsl.program_has_no_holes,
            max_depth=max_depth,
        )
        # succeed if this raises StopIteration
        with pytest.raises(StopIteration):
            n_iter = 0
            iterator = ns.search.bounded_astar_async(
                g,
                near.ValidationCost(
                    neural_dsl=neural_dsl,
                    trainer_cfg=trainer_cfg,
                    datamodule=datamodule,
                ),
                max_depth=max_depth,
                max_workers=4,
            )
            while True:
                print("iteration: ", n_iter)
                n_iter += 1
                node = next(iterator)
                self.assertIsNotNone(node)
                if n_iter > 30:
                    break
