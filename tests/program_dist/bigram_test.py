import unittest
from fractions import Fraction

import numpy as np
import torch

import neurosym as ns

from .utils import ProbabilityTester


def get_dsl():
    dslf = ns.DSLFactory()
    dslf.concrete("+", "(i, i) -> i", lambda x, y: x + y)
    dslf.concrete("1", "() -> i", lambda: 1)
    dslf.concrete("2", "() -> i", lambda: 2)
    dslf.prune_to("i")
    return dslf.finalize()


dsl = get_dsl()
fam = ns.BigramProgramDistributionFamily(dsl)

uniform = [
    [
        # root -> each
        [0, 1 / 3, 1 / 3, 1 / 3],
        [0] * 4,
    ],
    [
        # + -> each first arg
        [0, 1 / 3, 1 / 3, 1 / 3],
        # + -> each second arg
        [0, 1 / 3, 1 / 3, 1 / 3],
    ],
    [[0] * 4] * 2,
    [[0] * 4] * 2,
]


class BigramSamplerTest(ProbabilityTester):
    def test_uniform_sample_counts(self):
        n = 10_000
        pdelta = 0.015
        dist = fam.uniform()
        self.assertSameProbDist(dist, uniform)
        samples = fam.sample(dist, n, np.random.RandomState(0))
        samples = [ns.render_s_expression(x) for x in samples]
        self.assertBinomial(n, 1 / 3, pdelta, samples.count("(1)"))
        for plus_two_things in (
            "(+ (1) (1))",
            "(+ (1) (2))",
            "(+ (2) (1))",
            "(+ (2) (2))",
        ):
            self.assertBinomial(
                n, 1 / 3 * 1 / 3 * 1 / 3, pdelta, samples.count(plus_two_things)
            )


class BigramParameterShapeTest(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(fam.parameters_shape(), (4, 2, 4))


class BigramWithParametersGetParametersTest(ProbabilityTester):
    def test_produces_uniform_with_fixed_input(self):
        parameters = torch.zeros((4, 2, 4))
        for off in -1, 298, 29, 3:
            self.assertSameProbDist(
                fam.with_parameters((parameters + off)[None])[0], uniform
            )

    def test_irrelevant_entries_dont_matter(self):
        parameters = torch.zeros((4, 2, 4))
        parameters[0, 0, 0] = 1038
        print(parameters)
        self.assertSameProbDist(fam.with_parameters(parameters[None])[0], uniform)

        parameters = torch.zeros((4, 2, 4))
        parameters[0, 1, 1] = -3
        self.assertSameProbDist(fam.with_parameters(parameters[None])[0], uniform)

        parameters = torch.zeros((4, 2, 4))
        parameters[2] = 3
        self.assertSameProbDist(fam.with_parameters(parameters[None])[0], uniform)

    def test_can_make_deterministic(self):
        parameters = torch.zeros((4, 2, 4))
        # + -> always 1 then 2
        parameters[1, 0, 2] = 100
        parameters[1, 1, 3] = 100

        dist = fam.with_parameters(parameters[None])[0]
        n = 10_000
        samples = fam.sample(dist, n, np.random.RandomState(0))
        samples = [ns.render_s_expression(x) for x in samples]
        self.assertEqual(set(samples), {"(1)", "(2)", "(+ (1) (2))"})

        self.assertBinomial(n, 1 / 3, 0.015, samples.count("(1)"))
        self.assertBinomial(n, 1 / 3, 0.015, samples.count("(2)"))
        self.assertBinomial(n, 1 / 3, 0.015, samples.count("(+ (1) (2))"))

    def test_can_make_impossible(self):
        parameters = torch.zeros((4, 2, 4))
        # disable root -> +
        parameters[0, 0, 1] = -100

        dist = fam.with_parameters(parameters[None])[0]
        n = 10_000
        samples = fam.sample(dist, n, np.random.RandomState(0))
        samples = [ns.render_s_expression(x) for x in samples]
        self.assertEqual(set(samples), {"(1)", "(2)"})

        self.assertBinomial(n, 1 / 2, 0.015, samples.count("(1)"))
        self.assertBinomial(n, 1 / 2, 0.015, samples.count("(2)"))

    def test_can_weight_precisely(self):
        parameters = torch.zeros((4, 2, 4))
        # disable root -> +
        parameters[0, 0, 1] = -100
        # make root -> 1 twice as likely as root -> 2
        parameters[0, 0, 2] = np.log(2)

        dist = fam.with_parameters(parameters[None])[0]
        n = 10_000
        samples = fam.sample(dist, n, np.random.RandomState(0))
        samples = [ns.render_s_expression(x) for x in samples]
        self.assertEqual(set(samples), {"(1)", "(2)"})

        self.assertBinomial(n, 2 / 3, 0.015, samples.count("(1)"))
        self.assertBinomial(n, 1 / 3, 0.015, samples.count("(2)"))


class BigramCountProgramsTest(unittest.TestCase):
    def test_counts_single_program(self):
        data = [[ns.parse_s_expression("(+ (1) (2))")]]
        [counts] = fam.count_programs(data).counts
        np.testing.assert_equal(
            counts.cpu().numpy(),
            [
                [
                    # root -> +
                    [0, 1, 0, 0],
                    [0] * 4,
                ],
                [
                    # + -> 1 as the first arg
                    [0, 0, 1, 0],
                    # + -> 2 as the second arg
                    [0, 0, 0, 1],
                ],
                [[0] * 4] * 2,
                [[0] * 4] * 2,
            ],
        )

    def test_counts_multiple_programs(self):
        data = [
            [ns.parse_s_expression(x) for x in ("(+ (1) (2))", "(+ (1) (1))")],
        ]
        [counts] = fam.count_programs(data).counts
        np.testing.assert_equal(
            counts.cpu().numpy(),
            [
                [
                    # root -> +
                    [0, 2, 0, 0],
                    [0] * 4,
                ],
                [
                    # + -> first arg
                    [0, 0, 2, 0],
                    # + -> second arg
                    [0, 0, 1, 1],
                ],
                [[0] * 4] * 2,
                [[0] * 4] * 2,
            ],
        )

    def test_counts_separate_programs(self):
        data = [
            [ns.parse_s_expression("(+ (1) (2))")],
            [ns.parse_s_expression("(+ (1) (1))")],
        ]
        counts = fam.count_programs(data).counts
        np.testing.assert_equal(
            counts.cpu().numpy(),
            [
                [
                    [
                        # root -> +
                        [0, 1, 0, 0],
                        [0] * 4,
                    ],
                    [
                        # + -> first arg
                        [0, 0, 1, 0],
                        # + -> second arg
                        [0, 0, 0, 1],
                    ],
                    [[0] * 4] * 2,
                    [[0] * 4] * 2,
                ],
                [
                    [
                        # root -> +
                        [0, 1, 0, 0],
                        [0] * 4,
                    ],
                    [
                        # + -> first arg
                        [0, 0, 1, 0],
                        # + -> second arg
                        [0, 0, 1, 0],
                    ],
                    [[0] * 4] * 2,
                    [[0] * 4] * 2,
                ],
            ],
        )


class BigramParameterDifferenceLossTest(unittest.TestCase):
    def computeLoss(self, logits, programs):
        programs = [[ns.parse_s_expression(x) for x in ps] for ps in programs]
        print(programs)
        loss = fam.parameter_difference_loss(
            logits,
            fam.count_programs(programs),
        )
        loss = loss.detach().cpu().numpy()
        return loss

    def assertLoss(self, logits, programs, target):
        np.testing.assert_almost_equal(self.computeLoss(logits, programs), target)

    def assertUniformLoss(self, programs, target):
        logits = torch.zeros((1, 4, 2, 4))
        self.assertLoss(logits, programs, target)

    def test_leaf_program_logit(self):
        self.assertUniformLoss([["(1)"]], [np.log(3)])
        self.assertUniformLoss([["(2)"]], [np.log(3)])

    def test_multiple_program_logit(self):
        self.assertUniformLoss([["(1)", "(2)"]], [np.log(3) * 2])
        self.assertUniformLoss([["(1)", "(1)"]], [np.log(3) * 2])

    def test_separate_program_logit(self):
        self.assertUniformLoss([["(1)"], ["(2)"]], [np.log(3)] * 2)
        self.assertUniformLoss([["(1)"], ["(1)"]], [np.log(3)] * 2)

    def test_nested_program_logit(self):
        self.assertUniformLoss([["(+ (1) (2))"]], [np.log(3) * 3])

    def test_nonuniform_loss(self):
        logits = torch.zeros((2, 4, 2, 4))
        logits[0, 0, 0, 1] = np.log(3)  # 1/2 chance of +
        logits[0, 0, 0, 2] = np.log(2)  # 1/3 chance of 1, 1/6 chance of 2
        self.assertLoss(logits, [["(1)"], ["(1)"]], [-np.log(1 / 3), -np.log(1 / 3)])
        self.assertLoss(logits, [["(2)"], ["(2)"]], [-np.log(1 / 6), -np.log(1 / 3)])
        self.assertLoss(
            logits,
            [["(+ (1) (2))"], ["(+ (1) (2))"]],
            [-(np.log(1 / 2) + np.log(1 / 3) * 2), -3 * np.log(1 / 3)],
        )


class BigramLikelihoodTest(unittest.TestCase):
    def assertLikelihood(self, dist, program, str_prob):
        likelihood = fam.compute_likelihood(dist, ns.parse_s_expression(program))
        prob = np.exp(likelihood)
        prob = Fraction.from_float(float(prob)).limit_denominator(1000)
        result = f"log({prob})"
        print(result)
        self.assertEqual(result, str_prob)

    def test_leaf(self):
        self.assertLikelihood(fam.uniform(), "(1)", "log(1/3)")

    def test_plus(self):
        self.assertLikelihood(fam.uniform(), "(+ (1) (2))", "log(1/27)")

    def test_plus_nested(self):
        self.assertLikelihood(fam.uniform(), "(+ (1) (+ (1) (2)))", "log(1/243)")
