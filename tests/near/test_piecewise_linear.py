import itertools
import unittest

import numpy as np
from torch import nn

from parameterized import parameterized

import neurosym as ns
from neurosym.examples import near
from neurosym.examples.near.heirarchical.heirarchical_near import (
    heirarchical_near_graph,
)


def linear_replacement_dsl():
    dslf = ns.DSLFactory()
    dslf.parameterized(
        "aff_x",
        "{f, 2} -> {f, 1}",
        lambda x, aff: aff(x[:, 0][:, None]),
        dict(aff=lambda: nn.Linear(1, 1)),
    )
    dslf.parameterized(
        "aff_xplusy",
        "{f, 2} -> {f, 1}",
        lambda xy, aff: aff(xy[:, 0][:, None] + xy[:, 1][:, None]),
        dict(aff=lambda: nn.Linear(1, 1)),
    )
    dslf.parameterized(
        "aff_yminusx",
        "{f, 2} -> {f, 1}",
        lambda xy, aff: aff(xy[:, 1][:, None] - xy[:, 0][:, None]),
        dict(aff=lambda: nn.Linear(1, 1)),
    )
    dslf.lambdas()
    dslf.prune_to("{f, 2} -> {f, 1}")
    return dslf.finalize()


def piecewise_linear_dsl(linear_layers=True):
    dslf = ns.DSLFactory()
    # dslf.parameterized(
    #     "linear", "() -> {f, 2} -> {f, 2}", lambda lin: lin, dict(lin=nn.Linear(2, 2))
    # )
    if linear_layers:
        dslf.parameterized(
            "linear_bool",
            "() -> {f, 2} -> {f, 1}",
            lambda lin: lin,
            dict(lin=lambda: nn.Linear(2, 1)),
        )
    else:
        dslf.parameterized(
            "aff_x",
            "{f, 2} -> {f, 1}",
            lambda x, aff: aff(x[:, 0][:, None]),
            dict(aff=lambda: nn.Linear(1, 1)),
        )
        dslf.concrete(
            "xplusy",
            "{f, 2} -> {f, 1}",
            lambda xy: xy[:, 0][:, None] + xy[:, 1][:, None],
        )
        dslf.concrete(
            "yminusx",
            "{f, 2} -> {f, 1}",
            lambda xy: xy[:, 1][:, None] - xy[:, 0][:, None],
        )
        dslf.lambdas()

    dslf.concrete(
        "ite",
        "(#a -> {f, 1}, #a -> #b, #a -> #b) -> #a -> #b",
        near.operations.ite_torch,
    )

    dslf.prune_to("{f, 2} -> {f, 1}")
    return dslf.finalize()


def get_dataset():
    """
    Function in this dataset is a piecewise linear function

    f(x, y) = x + y if x > 0
    f(x, y) = -x + y if x <= 0

    or in other words

    f(x, y) = |x| + y
    """
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


neural_hole_filler = near.GenericMLPRNNNeuralHoleFiller(hidden_size=10)


def get_neural_dsl(dsl):
    return near.NeuralDSL.from_dsl(
        dsl=dsl,
        neural_hole_filler=neural_hole_filler,
    )


def get_validation_cost(dsl, dataset, **validation_params):
    neural_dsl = get_neural_dsl(dsl)
    return near.ValidationCost(
        trainer_cfg=near.NEARTrainerConfig(
            lr=0.005,
            n_epochs=100,
            accelerator="cpu",
            loss_callback=nn.functional.mse_loss,
        ),
        neural_dsl=neural_dsl,
        datamodule=dataset,
        progress_by_epoch=False,
        **validation_params,
    )


class TestPiecewiseLinear(unittest.TestCase):

    def near_graph(self, neural_dsl, validation_cost):

        return near.validated_near_graph(
            neural_dsl,
            ns.parse_type("{f, 2} -> {f, 1}"),
            is_goal=lambda _: True,
            max_depth=10000,
            cost=validation_cost,
            validation_epochs=1000,
        )

    def search(self, g, count=3, max_iters=10):

        iterator = ns.search.bounded_astar(g, max_depth=10000, max_iterations=max_iters)

        return list(itertools.islice(iterator, count))

    def test_with_linear(self):
        dsl = piecewise_linear_dsl()
        dataset = get_dataset()
        result = self.search(
            self.near_graph(get_neural_dsl(dsl), get_validation_cost(dsl, dataset))
        )

        programs = [
            ns.render_s_expression(p.initalized_program.uninitialize()) for p in result
        ]

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

    def grab_desired(self, result):
        programs = [
            ns.render_s_expression(p.initalized_program.uninitialize()) for p in result
        ]

        expected = "(ite (linear_bool) (linear_bool) (linear_bool))"
        self.assertIn(expected, programs)

        return result[programs.index(expected)]

    def test_with_variables(self):
        dsl = piecewise_linear_dsl(linear_layers=False)
        dataset = get_dataset()
        result = self.search(
            self.near_graph(get_neural_dsl(dsl), get_validation_cost(dsl, dataset))
        )
        self.assertEqual(result, [])

    @parameterized.expand([() for _ in range(10)])
    def test_heirarchical(self):
        l_dsl = piecewise_linear_dsl()
        lr_dsl = linear_replacement_dsl()
        g = heirarchical_near_graph(
            l_dsl,
            "linear_bool",
            lr_dsl,
            ns.parse_type("{f, 2} -> {f, 1}"),
            lambda dsl, embedding: get_validation_cost(
                dsl, get_dataset(), embedding=embedding
            ),
            neural_hole_filler,
            is_goal=lambda _: True,
            max_depth=10000,
            validation_epochs=1000,
        )
        results = self.search(g, 1, 1000)
        self.assertEqual(len(results), 1)
        result = results[0]
        program_str = ns.render_s_expression(result.uninitialize())
        negative = "(ite (lam (aff_x ($0_0))) (lam (aff_yminusx ($0_0))) (lam (aff_xplusy ($0_0))))"
        positive = "(ite (lam (aff_x ($0_0))) (lam (aff_xplusy ($0_0))) (lam (aff_yminusx ($0_0))))"
        self.assertIn(program_str, [negative, positive])
        children = [s.children[0].state["aff"] for s in results[0].children]
        cond, cons, alt = [(s.weight.item(), s.bias.item()) for s in children]
        if program_str == negative:
            cond = -cond[0], -cond[1]

        self.assertGreater(cond[0], 1)
        self.assertLess(np.abs(cond[1]), 0.25)

        for branch in cons, alt:
            print(branch)
            self.assertLess(np.abs(+1 - branch[0]), 0.25)
            self.assertLess(np.abs(branch[1]), 0.25)
