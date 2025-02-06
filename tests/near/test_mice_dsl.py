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
    @staticmethod
    def tinycrim13_binary_cross_entropy_loss(
        predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the binary cross entropy loss with class weights for the Tiny CRIM13 dataset.
        This is the same loss function used in the base NEAR implementation.
        predictions: (B, T, O)
        targets: (B, T, 1)
        """
        targets = targets.squeeze(-1)  # (B, T, 1) -> (B, T)
        predictions = predictions.view(-1, predictions.shape[-1])
        targets = targets.view(-1)
        targets_one_hot = torch.nn.functional.one_hot(
            targets, num_classes=2
        )  # pylint: disable=not-callable
        return torch.nn.functional.binary_cross_entropy_with_logits(
            predictions,
            targets_one_hot.float(),
            weight=torch.tensor([2.0, 1.0], device=predictions.device),
        )

    def test_replicate_tinycrim13(self):
        """
        Ensure that the performance of the program is atleast 90% of the performance of the base NEAR implementation.
        """
        datamodule = ns.datasets.crim13_investigation_example(
            train_seed=0, batch_size=1024
        )
        _, output_dim = datamodule.train.get_io_dims()
        original_dsl = near.simple_crim13_dsl(num_classes=output_dim, hidden_dim=16)
        trainer_cfg = near.NEARTrainerConfig(
            n_epochs=12,
            lr=1e-4,
            loss_callback=self.tinycrim13_binary_cross_entropy_loss,
        )
        neural_dsl = near.NeuralDSL.from_dsl(
            dsl=original_dsl,
            neural_hole_filler=near.GenericMLPRNNNeuralHoleFiller(hidden_size=16),
        )
        # structural cost goes from 0 -> \inf, each delta is around +/- 2.
        # validation cost goes from 0 -> 1, each delta is around +/- 0.1.
        # To ensure equal weight, we scale the structural cost by 0.05 so that
        # the delta is around 2 * 0.05 = +/-0.1.
        cost = near.default_near_cost(
            trainer_cfg=trainer_cfg,
            datamodule=datamodule,
        )

        g = near.near_graph(
            neural_dsl,
            neural_dsl.valid_root_types[0],
            is_goal=lambda _: True,
            cost=cost,
        )
        iterator = ns.search.bounded_astar(
            g,
            max_depth=5,
        )
        # Should not throw a StopIteration error
        program = next(iterator)
        initialized_program = neural_dsl.initialize(program)
        _ = cost.validation_heuristic.with_n_epochs(
            40
        ).compute_cost(  # pylint: disable=no-member
            neural_dsl, initialized_program, cost.embedding
        )
        self.assertIsNotNone(initialized_program)
        # ensure that the node's F1 score is within 0.1 of the base NEAR implementation 0.8 F1 score.
        feature_data = datamodule.test.inputs
        labels = datamodule.test.outputs.flatten()
        # module, _ = validation_cost.validate_model(best_program, n_epochs=15)
        module = ns.examples.near.TorchProgramModule(neural_dsl, initialized_program)
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
