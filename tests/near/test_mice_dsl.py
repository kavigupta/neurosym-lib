"""
Test CRIM13 DSL with NEAR search graph.

We conduct the following tests:
- Sanity check: We can find a simple program.
- Performance Check: Our program performs as well as the base NEAR implementation.
"""

import unittest

import torch
from sklearn.metrics import classification_report

import neurosym as ns
from neurosym.examples import near

from .utils import assertDSLEnumerable


class TestNEARMiceDSL(unittest.TestCase):
    def test_mice_dsl_perf(self):
        """
        Ensure that the performance of the program is atleast 90% of the performance of the base NEAR implementation.
        """
        datamodule = ns.datasets.crim13_investigation_example(
            train_seed=0, batch_size=1024
        )
        _, output_dim = datamodule.train.get_io_dims()
        original_dsl = near.simple_crim13_dsl(num_classes=output_dim, hidden_dim=10)
        trainer_cfg = near.NEARTrainerConfig(n_epochs=10, lr=0.01)
        neural_dsl = near.NeuralDSL.from_dsl(
            dsl=original_dsl,
            neural_hole_filler=near.GenericMLPRNNNeuralHoleFiller(hidden_size=10),
        )
        validation_cost = near.ValidationCost(
            neural_dsl=neural_dsl,
            trainer_cfg=trainer_cfg,
            datamodule=datamodule,
        )

        g = near.near_graph(
            neural_dsl,
            neural_dsl.valid_root_types[0],
            is_goal=lambda _: True,
            cost=validation_cost,
        )
        iterator = ns.search.bounded_astar(
            g,
            max_depth=4,
        )
        # Should not throw a StopIteration error
        best_program = next(iterator)
        self.assertIsNotNone(best_program)

        # ensure that the node's F1 score is within 0.1 of the base NEAR implementation 0.8 F1 score.
        feature_data = datamodule.test.inputs
        labels = datamodule.test.outputs.flatten()
        module, _ = validation_cost.validate_model(best_program, n_epochs=15)
        predictions = (
            module(torch.tensor(feature_data), environment=()).argmax(-1).numpy()
        )
        report = classification_report(
            labels,
            predictions,
            target_names=["not investigation", "investigation"],
            output_dict=True,
        )
        self.assertGreaterEqual(
            report["not investigation"]["f1-score"], 0.76855895 * 0.9
        )
        self.assertGreaterEqual(report["investigation"]["f1-score"], 0.57237779 * 0.9)

    def test_mice_dsl_enumerate(self):
        """
        Enumerate all programs in dsl upto fixed depth. This test case makes
        sure all DSL combinations upto a fixed depth are valid.
        """
        self.maxDiff = None
        dsl = near.simple_crim13_dsl(num_classes=2)

        assertDSLEnumerable(dsl, "([$fL]) -> $fO")
