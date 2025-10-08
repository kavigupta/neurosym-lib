import itertools
import unittest

from torch import nn

import neurosym as ns
from neurosym.examples import near

dataset_factory = lambda train_seed: ns.DatasetWrapper(
    ns.DatasetFromNpy(
        "tutorial/bouncing_ball_exercise/data/bounce_example/train_ex_data.npy",
        "tutorial/bouncing_ball_exercise/data/bounce_example/train_ex_labels.npy",
        train_seed,
    ),
    ns.DatasetFromNpy(
        "tutorial/bouncing_ball_exercise/data/bounce_example/test_ex_data.npy",
        "tutorial/bouncing_ball_exercise/data/bounce_example/test_ex_labels.npy",
        None,
    ),
    batch_size=200,
)

L = 4


def bounce_dsl():
    dslf = ns.DSLFactory(L=L, max_overall_depth=5)
    # BEGIN SOLUTION "YOUR CODE HERE"
    ## DSL for the bounce example.
    dslf.typedef("fL", "{f, $L}")

    dslf.production(
        "linear_bool",
        "() -> $fL -> {f, 1}",
        lambda lin: lin,
        dict(lin=lambda: nn.Linear(L, 1)),
    )
    dslf.production(
        "linear", "() -> $fL -> $fL", lambda lin: lin, dict(lin=lambda: nn.Linear(L, L))
    )

    dslf.production(
        "ite",
        "(#a -> {f, 1}, #a -> #a, #a -> #a) -> #a -> #a",
        near.operations.ite_torch,
    )
    dslf.production(
        "map",
        "(#a -> #b) -> [#a] -> [#b]",
        lambda f: lambda x: near.operations.map_torch(f, x),
    )
    # END SOLUTION
    dslf.prune_to("[$fL] -> [$fL]")
    return dslf.finalize()


def predicate_dsl():
    dslf = ns.DSLFactory(L=L, max_overall_depth=5)
    dslf.typedef("fL", "{f, $L}")

    def add_projection(name, channel):
        dslf.production(
            name,
            "$fL -> {f, 1}",
            lambda x, lin: lin(x[..., [channel]]),
            dict(lin=lambda: nn.Linear(1, 1)),
        )

    for channel, name in enumerate(["aff_x", "aff_y", "aff_vx", "aff_vy"]):
        add_projection(name, channel)
    dslf.lambdas()
    dslf.prune_to("$fL -> {f, 1}")
    return dslf.finalize()


class TestHierarchicalBouncingBall(unittest.TestCase):

    def check_ground_bounce(self, node):
        cond, cons, alt = node.children
        cond = cond.children[0].state["lin"]
        w, b = cond.weight.item(), cond.bias.item()
        self.assertGreater(abs(w), 10)
        self.assertLess(abs(b), 1)
        bounce = cons if w < 0 else alt
        bounce_vy_vy = bounce.state["lin"].weight[-1, -1].item()
        print(bounce_vy_vy)
        self.assertLess(bounce_vy_vy, 0)

    def test_heirarchical_bouncing_ball(self):
        filler = near.GenericMLPRNNNeuralHoleFiller(hidden_size=10)

        g = near.heirarchical_near_graph(
            high_level_dsl=bounce_dsl(),
            symbol="linear_bool",
            refined_dsl=predicate_dsl(),
            typ=ns.parse_type("([{f, 4}]) -> [{f, 4}]"),
            validation_cost_creator=lambda embedding, symbol_costs: near.default_near_cost(
                trainer_cfg=near.NEARTrainerConfig(
                    lr=0.1,
                    n_epochs=100,
                    accelerator="cpu",
                    loss_callback=nn.functional.mse_loss,
                ),
                datamodule=dataset_factory(42),
                progress_by_epoch=True,
                embedding=embedding,
                # structural_cost_weight=0.2,
                symbol_costs=symbol_costs,
            ),
            neural_hole_filler=filler,
            validation_epochs=4000,
        )
        best_programs = itertools.islice(
            ns.search.bounded_astar(g, max_depth=1000, max_iterations=10000), 4
        )
        num_with_one_ground_bounce_branch = 0
        for prog in best_programs:
            print(ns.render_s_expression(prog.uninitialize()))
            ground_bounce_branching = [
                node
                for node in ns.postorder(prog)
                if node.symbol == "ite"
                and node.children
                and node.children[0].children
                and node.children[0].children[0].symbol == "aff_y"
            ]
            if len(ground_bounce_branching) != 1:
                continue
            self.check_ground_bounce(ground_bounce_branching[0])
            num_with_one_ground_bounce_branch += 1
        self.assertGreater(num_with_one_ground_bounce_branch, 0)
