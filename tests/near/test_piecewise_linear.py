import unittest

import numpy as np
from torch import nn

import neurosym as ns
from neurosym.examples import near


def piecewise_linear_dsl():
    dslf = ns.DSLFactory()
    # dslf.parameterized(
    #     "linear", "() -> {f, 2} -> {f, 2}", lambda lin: lin, dict(lin=nn.Linear(2, 2))
    # )
    dslf.parameterized(
        "linear_bool",
        "() -> {f, 2} -> {f, 1}",
        lambda lin: lin,
        dict(lin=lambda: nn.Linear(2, 1)),
    )
    dslf.concrete(
        "ite",
        "(#a -> {f, 1}, #a -> #b, #a -> #b) -> #a -> #b",
        near.operations.ite_torch,
    )

    dslf.prune_to("{f, 2} -> {f, 1}")
    return dslf.finalize()


def get_dataset():
    scale = 2
    train_size, test_size = 1000, 50
    rng = np.random.RandomState(0)
    xtrain, xtest = [
        scale * rng.randn(size, 2).astype(np.float32)
        for size in (train_size, test_size)
    ]
    ytrain, ytest = [
        # x[:, 1][:, None] for x in (xtrain, xtest)
        ((x[:, 0] > 0) * (x[:, 0] + x[:, 1]) + (x[:, 0] <= 0) * (-x[:, 0] + x[:, 1]))[
            :, None
        ].astype(np.float32)
        for x in (xtrain, xtest)
    ]
    return ns.DatasetWrapper(
        ns.DatasetFromNpy(xtrain, ytrain, None),
        ns.DatasetFromNpy(xtest, ytest, None),
        batch_size=100,
    )


class TestPiecewiseLinear(unittest.TestCase):
    def test_basic(self):
        dsl = piecewise_linear_dsl()

        dataset = get_dataset()
        interface = near.NEAR(
            input_dim=2,
            output_dim=1,
            max_depth=10000,
            # lr hilariously high and n_epochs hilariously low but it's fine
            # for what we're investigating since the wrong_i simply do not work
            lr=0.005,
            max_seq_len=300,
            n_epochs=100,
            accelerator="cpu",
        )

        interface.register_search_params(
            dsl=dsl,
            type_env=ns.TypeDefiner(),
            neural_hole_filler=near.GenericMLPRNNNeuralHoleFiller(hidden_size=10),
            search_strategy=ns.search.bounded_astar,
            loss_callback=nn.functional.mse_loss,
            validation_params=dict(
                enable_progress_bar=False,
                enable_model_summary=False,
                progress_by_epoch=False,
            ),
        )

        result = interface.fit(
            datamodule=dataset,
            program_signature="{f, 2} -> {f, 1}",
            n_programs=3,
            validation_max_epochs=1000,
            max_iterations=10,
        )

        programs = [ns.render_s_expression(p.program) for p in result]

        expected = "(ite (linear_bool) (linear_bool) (linear_bool))"
        self.assertIn(expected, programs)

        result_relevant = result[programs.index(expected)]

        [cond, cons, alt] = [
            x.state["lin"] for x in result_relevant.initalized_program.children
        ]

        print("Conditional weight:")
        print(cond.weight)
        print("Consequent weight and bias:")
        print(cons.weight)
        print(cons.bias)
        print("Alternative weight and bias:")
        print(alt.weight)
        print(alt.bias)

        conditional_weight = cond.weight[0, 0].item()

        self.assertGreater(np.abs(conditional_weight), 1)
        self.assertLess(np.abs(cond.weight[0, 1].item()), 1)

        if conditional_weight < 0:
            cons, alt = alt, cons

        self.assertLess(np.abs(+1 - cons.weight[0, 0].item()), 0.1)
        self.assertLess(np.abs(+1 - cons.weight[0, 1].item()), 0.1)

        self.assertLess(np.abs(-1 - alt.weight[0, 0].item()), 0.1)
        self.assertLess(np.abs(+1 - alt.weight[0, 1].item()), 0.1)
