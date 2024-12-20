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
        heuristic = validation score after training for N epochs.
        goal = Fully symbolic program. (handled in: search_graph/dsl_search_graph.py)
        test_predicate = score on testing set

        This tests an async version of bounded_astar search.
        """
        # pylint: disable=duplicate-code
        datamodule = ns.datasets.near_data_example(
            train_seed=0, batch_size=32, num_workers=0
        )
        input_dim, output_dim = datamodule.train.get_io_dims()
        original_dsl = near.example_rnn_dsl(input_dim, output_dim)
        trainer_cfg = near.NEARTrainerConfig(n_epochs=10)
        t = ns.TypeDefiner(L=input_dim, O=output_dim)
        t.typedef("fL", "{f, $L}")
        t.typedef("fO", "{f, $O}")
        neural_dsl = near.NeuralDSL.from_dsl(
            dsl=original_dsl,
            neural_hole_filler=near.GenericMLPRNNNeuralHoleFiller(hidden_size=10),
        )
        max_depth = 3

        g = near.near_graph(
            neural_dsl,
            ns.parse_type(
                s="([{f, $L}]) -> [{f, $O}]",
                env=ns.TypeDefiner(L=input_dim, O=output_dim),
            ),
            is_goal=lambda _: True,
            max_depth=max_depth,
            cost=near.default_near_cost(
                trainer_cfg=trainer_cfg,
                datamodule=datamodule,
            ),
        )
        # succeed if this raises StopIteration
        with pytest.raises(StopIteration):
            n_iter = 0
            iterator = ns.search.bounded_astar_async(
                g,
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
