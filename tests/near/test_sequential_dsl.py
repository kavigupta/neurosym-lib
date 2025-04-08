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
        heuristic = validation score after training for N epochs.
        goal = Fully symbolic program. (handled in: search_graph/dsl_search_graph.py)
        test_predicate = score on testing set
        """
        datamodule = ns.datasets.near_data_example(train_seed=0)
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

        g = near.near_graph(
            neural_dsl,
            ns.parse_type(
                s="([{f, $L}]) -> [{f, $O}]",
                env=ns.TypeDefiner(L=input_dim, O=output_dim),
            ),
            is_goal=lambda _: True,
            cost=near.default_near_cost(
                trainer_cfg=trainer_cfg,
                datamodule=datamodule,
            ),
        )
        # succeed if this raises StopIteration
        with pytest.raises(StopIteration):
            n_iter = 0
            iterator = ns.search.bounded_astar(
                g,
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
