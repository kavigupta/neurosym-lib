"""
Check that the heuristic is actually being used in neural model search, and that
it works as expected.
"""

import unittest

import numpy as np
import torch

import neurosym as ns
from neurosym.examples import near


def dsl_with_scalars(typ):
    def wrap(f):
        if typ == "{f, 1}":
            return f
        assert typ == "f"
        return lambda x: f(x[:, 0])[:, None]

    dslf = ns.DSLFactory()
    dslf.typedef("f", typ)
    dslf.production("zero", "$f -> $f", lambda x: 0 * x)
    dslf.production("double", "$f -> $f", lambda x: 2 * x)
    if typ != "{f, 1}":
        dslf.production("wrap", "($f -> $f) -> {f, 1} -> {f, 1}", wrap)
    dslf.lambdas()
    dslf.prune_to("{f, 1} -> {f, 1}")
    return dslf.finalize()


def datamodule(multiplier):
    rng = np.random.default_rng(0)
    xtrain, xtest = rng.normal(size=(1000, 1)).astype(np.float32), rng.normal(
        size=(50, 1)
    ).astype(np.float32)
    ytrain, ytest = multiplier * xtrain, multiplier * xtest
    return ns.DatasetWrapper(
        ns.DatasetFromNpy(xtrain, ytrain, None),
        ns.DatasetFromNpy(xtest, ytest, None),
        batch_size=100,
    )


class TestHeuristicUsedScalars(unittest.TestCase):

    def run_model(self, dsl, multiplier, hole_filler):
        interface = near.NEAR(
            max_depth=100000,
            lr=0.01,
            n_epochs=50,
            accelerator="cpu",
        )
        dm = datamodule(multiplier)

        def check(node):
            f2 = dsl.compute(dsl.initialize(node.program))
            delta = ((f2(dm.train.inputs) - dm.train.outputs) ** 2).mean()
            return delta.item() < 1e-3

        interface.register_search_params(
            dsl=dsl,
            type_env=ns.TypeDefiner(),
            neural_hole_filler=hole_filler,
            search_strategy=ns.search.BoundedAStar(
                max_depth=float("inf"), max_iterations=100
            ),
            loss_callback=torch.nn.functional.mse_loss,
            validation_params=dict(progress_by_epoch=False),
            validation_metric="neg_l2_dist",
            is_goal=check,
        )
        res = interface.fit(
            datamodule=dm,
            program_signature="{f, 1} -> {f, 1}",
            n_programs=1,
            validation_max_epochs=100,
        )
        res = [ns.render_s_expression(x) for x in res]
        print(res)
        return res

    def test_basic(self):
        dsl = dsl_with_scalars("{f, 1}")
        self.assertEqual(
            self.run_model(dsl, 0, near.GenericMLPRNNNeuralHoleFiller(100)),
            ["(lam (zero ($0_0)))"],
        )

    def test_multiple(self):
        dsl = dsl_with_scalars("{f, 1}")
        self.assertEqual(
            self.run_model(dsl, 2**20, near.GenericMLPRNNNeuralHoleFiller(100)),
            [
                "(lam (double (double (double (double (double (double (double (double (double (double (double (double (double (double (double (double (double (double (double (double ($0_0))))))))))))))))))))))"
            ],
        )

    def test_multiple_transformer(self):
        dsl = dsl_with_scalars("{f, 1}")
        self.assertEqual(
            self.run_model(
                dsl,
                2**5,
                near.TransformerNeuralHoleFiller(
                    max_tensor_size=100,
                    hidden_size=8,
                    num_decoder_layers=1,
                    num_encoder_layers=1,
                    num_head=2,
                ),
            ),
            ["(lam (double (double (double (double (double ($0_0)))))))"],
        )

    def test_multiple_no_hole_filler(self):
        dsl = dsl_with_scalars("{f, 1}")
        self.assertEqual(
            self.run_model(dsl, 2**7, near.DoNothingNeuralHoleFiller()),
            [],
        )

    def test_basic_scalars(self):
        dsl = dsl_with_scalars("f")
        self.assertEqual(
            self.run_model(dsl, 0, near.GenericMLPRNNNeuralHoleFiller(100)),
            ["(wrap (lam_0 (zero ($0_0))))"],
        )

    def test_multiple_scalars(self):
        dsl = dsl_with_scalars("f")
        self.assertEqual(
            self.run_model(dsl, 2**32, near.GenericMLPRNNNeuralHoleFiller(100)),
            [
                "(wrap (lam_0 (double (double (double (double (double (double (double (double (double (double (double (double (double (double (double (double (double (double (double (double (double (double (double (double (double (double (double (double (double (double (double (double ($0_0)))))))))))))))))))))))))))))))))))"
            ],
        )

    def test_multiple_scalars_transformer(self):
        dsl = dsl_with_scalars("f")
        self.assertEqual(
            self.run_model(
                dsl,
                2**5,
                near.TransformerNeuralHoleFiller(
                    max_tensor_size=100,
                    hidden_size=8,
                    num_decoder_layers=1,
                    num_encoder_layers=1,
                    num_head=2,
                ),
            ),
            ["(wrap (lam_0 (double (double (double (double (double ($0_0))))))))"],
        )
