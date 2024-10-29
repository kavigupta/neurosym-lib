import unittest
from fractions import Fraction

import numpy as np
import torch
from parameterized import parameterized

import neurosym as ns
from tests.utils import assertDSL

from .utils import (
    ChildrenInOrderAsserterMask,
    ChildrenInOrderMask,
    ProbabilityTester,
    enumerate_dsl,
)


def get_dsl(with_vars=False, with_3=False):
    dslf = ns.DSLFactory()
    dslf.concrete("+", "(i, i) -> i", lambda x, y: x + y)
    dslf.concrete("1", "() -> i", lambda: 1)
    dslf.concrete("2", "() -> i", lambda: 2)
    if with_3:
        dslf.concrete("3", "() -> i", lambda: 3)
    if with_vars:
        dslf.concrete("call", "(i -> i, i) -> i", lambda f, x: f(x))
        dslf.lambdas()
    dslf.prune_to("i")
    return dslf.finalize()


def get_dsl_for_ordering():
    dslf = ns.DSLFactory()
    dslf.concrete("+", "(i, i, i) -> t", lambda x, y: x + y)
    dslf.concrete("1", "() -> i", lambda: 1)
    dslf.concrete("2", "() -> i", lambda: 2)
    dslf.concrete("3", "() -> i", lambda: 2)
    dslf.prune_to("t")
    return dslf.finalize()


def get_more_complex_dsl():
    dslf = ns.DSLFactory()
    dslf.concrete("0", "() -> i", lambda: 0)
    dslf.concrete("+", "(i, i) -> i", lambda x, y: x + y)
    dslf.concrete("*", "(f, f) -> f", lambda x, y: x + y)
    dslf.concrete("sqrt", "i -> f", np.sqrt)
    dslf.concrete(">", "(f, f) -> b", np.sqrt)
    dslf.concrete("<", "(i, i) -> b", np.sqrt)
    dslf.concrete("ite", "(b, f, f) -> f", lambda cond, x, y: x if cond else y)
    dslf.lambdas()
    dslf.prune_to("f -> f")
    return dslf.finalize()


dsl = get_dsl()
fam = ns.BigramProgramDistributionFamily(dsl)
dsl_with_3 = get_dsl(with_3=True)
fam_with_3 = ns.BigramProgramDistributionFamily(dsl_with_3)
fam_with_3_no_type_mask = ns.BigramProgramDistributionFamily(
    dsl_with_3, include_type_preorder_mask=False
)
dsl_with_vars = get_dsl(with_vars=True)
fam_with_vars = ns.BigramProgramDistributionFamily(dsl_with_vars)
dsl_for_ordering = get_dsl_for_ordering()
fam_with_ordering = ns.BigramProgramDistributionFamily(
    dsl_for_ordering, additional_preorder_masks=[ChildrenInOrderMask]
)
fam_with_ordering_asserted = ns.BigramProgramDistributionFamily(
    dsl_for_ordering, additional_preorder_masks=[ChildrenInOrderAsserterMask]
)
fam_with_ordering_231 = ns.BigramProgramDistributionFamily(
    dsl_for_ordering,
    additional_preorder_masks=[ChildrenInOrderMask],
    # 2 3 1 means go to index 2 first, then 0 second, then 1 third
    node_ordering=lambda dist: ns.DictionaryNodeOrdering(dist, {"+": [2, 0, 1]}),
)

dsl_more_complex = get_more_complex_dsl()
fam_more_complex = ns.BigramProgramDistributionFamily(dsl_more_complex)

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
        samples = [fam.sample(dist, np.random.RandomState(i)) for i in range(n)]
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
        samples = [fam.sample(dist, np.random.RandomState(i)) for i in range(n)]
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
        samples = [fam.sample(dist, np.random.RandomState(i)) for i in range(n)]
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
        samples = [fam.sample(dist, np.random.RandomState(i)) for i in range(n)]
        samples = [ns.render_s_expression(x) for x in samples]
        self.assertEqual(set(samples), {"(1)", "(2)"})

        self.assertBinomial(n, 2 / 3, 0.015, samples.count("(1)"))
        self.assertBinomial(n, 1 / 3, 0.015, samples.count("(2)"))

    def test_fam_with_variables_dsl(self):
        assertDSL(
            self,
            dsl_with_vars.render(),
            """
            $0_0 :: V<i@0>
            $1_0 :: V<i@1>
            $2_0 :: V<i@2>
            $3_0 :: V<i@3>
            + :: (i, i) -> i
            1 :: () -> i
            2 :: () -> i
            call :: (i -> i, i) -> i
            lam :: L<#body|i> -> i -> #body
            """,
        )

    def test_sample_with_variables(self):
        dist = fam_with_vars.uniform()
        n = 20000
        samples = [
            fam_with_vars.sample(dist, np.random.RandomState(i)) for i in range(n)
        ]
        samples = [ns.render_s_expression(x) for x in samples]
        self.assertBinomial(n, 1 / 4, 0.01, samples.count("(1)"))
        self.assertBinomial(n, 1 / 4, 0.01, samples.count("(2)"))
        # see test_call_with_variables for the math
        self.assertBinomial(n, 1 / 80, 0.01, samples.count("(call (lam ($0_0)) (1))"))

    def test_sample_with_ordering(self):
        dist = fam_with_ordering.uniform()
        n = 10_000
        samples = [
            fam_with_ordering.sample(dist, np.random.RandomState(i)) for i in range(n)
        ]
        samples = [ns.render_s_expression(x) for x in samples]
        self.assertEqual(set(samples), {"(+ (1) (2) (3))"})

    def test_sample_with_ordering_231(self):
        dist = fam_with_ordering_231.uniform()
        n = 10_000
        samples = [
            fam_with_ordering_231.sample(dist, np.random.RandomState(i))
            for i in range(n)
        ]
        samples = [ns.render_s_expression(x) for x in samples]
        self.assertEqual(set(samples), {"(+ (2) (3) (1))"})


class BigramCountProgramsTest(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def convert(self, family, count):
        symbol_text = lambda x: family.tree_distribution_skeleton.symbols[x][0]
        numerators = {
            tuple((symbol_text(sym), pos) for sym, pos in chain): {
                symbol_text(sym): count for sym, count in counts.items()
            }
            for chain, counts in count.numerators.items()
        }
        denominators = {
            tuple((symbol_text(sym), pos) for sym, pos in chain): {
                tuple(symbol_text(sym) for sym in syms): count
                for syms, count in counts.items()
            }
            for chain, counts in count.denominators.items()
        }
        return numerators, denominators

    def count_programs(self, family, programs):
        counts = family.count_programs(
            [[ns.parse_s_expression(x) for x in ps] for ps in programs]
        )
        return [self.convert(family, count) for count in counts.counts]

    def test_counts_single_program(self):
        counts = self.count_programs(fam, [["(+ (1) (2))"]])
        self.assertEqual(
            counts,
            [
                (
                    {
                        (("<root>", 0),): {"+": 1},
                        (("+", 0),): {"1": 1},
                        (("+", 1),): {"2": 1},
                    },
                    {
                        # all symbols could be anything else
                        (("<root>", 0),): {("+", "1", "2"): 1},
                        (("+", 0),): {("+", "1", "2"): 1},
                        (("+", 1),): {("+", "1", "2"): 1},
                    },
                )
            ],
        )

    def test_counts_multiple_programs(self):
        counts = self.count_programs(fam, [["(+ (1) (2))", "(+ (1) (1))"]])
        print(counts)

        self.assertEqual(
            counts,
            [
                (
                    {
                        (("<root>", 0),): {"+": 2},
                        (("+", 0),): {"1": 2},
                        (("+", 1),): {"2": 1, "1": 1},
                    },
                    {
                        (("<root>", 0),): {("+", "1", "2"): 2},
                        (("+", 0),): {("+", "1", "2"): 2},
                        (("+", 1),): {("+", "1", "2"): 2},
                    },
                )
            ],
        )

    def test_counts_separate_programs(self):
        counts = self.count_programs(fam, [["(+ (1) (2))"], ["(+ (1) (1))"]])
        self.assertEqual(
            counts,
            [
                (
                    {
                        (("<root>", 0),): {"+": 1},
                        (("+", 0),): {"1": 1},
                        (("+", 1),): {"2": 1},
                    },
                    {
                        (("<root>", 0),): {("+", "1", "2"): 1},
                        (("+", 0),): {("+", "1", "2"): 1},
                        (("+", 1),): {("+", "1", "2"): 1},
                    },
                ),
                (
                    {
                        (("<root>", 0),): {"+": 1},
                        (("+", 0),): {"1": 1},
                        (("+", 1),): {"1": 1},
                    },
                    {
                        (("<root>", 0),): {("+", "1", "2"): 1},
                        (("+", 0),): {("+", "1", "2"): 1},
                        (("+", 1),): {("+", "1", "2"): 1},
                    },
                ),
            ],
        )

    def test_counts_variables(self):
        counts = self.count_programs(fam_with_vars, [["(call (lam ($0_0)) (1))"]])
        self.assertEqual(
            counts,
            [
                (
                    {
                        (("<root>", 0),): {"call": 1},
                        (("call", 0),): {"lam": 1},
                        (("lam", 0),): {"$0_0": 1},
                        (("call", 1),): {"1": 1},
                    },
                    {
                        (("<root>", 0),): {("+", "1", "2", "call"): 1},
                        (("call", 0),): {("lam",): 1},
                        (("lam", 0),): {("$0_0", "+", "1", "2", "call"): 1},
                        (("call", 1),): {("+", "1", "2", "call"): 1},
                    },
                )
            ],
        )

    def test_counts_single_program_ordered(self):
        counts = self.count_programs(fam_with_ordering, [["(+ (1) (2) (3))"]])
        print(counts)
        self.assertEqual(
            counts,
            [
                (
                    {
                        (("<root>", 0),): {"+": 1},
                        (("+", 0),): {"1": 1},
                        (("+", 1),): {"2": 1},
                        (("+", 2),): {"3": 1},
                    },
                    {
                        # note that there's no multiplicity here
                        # because the ordering mask doesn't allow it
                        (("<root>", 0),): {("+",): 1},
                        (("+", 0),): {("1",): 1},
                        (("+", 1),): {("2",): 1},
                        (("+", 2),): {("3",): 1},
                    },
                )
            ],
        )

    def test_counts_single_program_ordered_231(self):
        counts = self.count_programs(fam_with_ordering_231, [["(+ (2) (3) (1))"]])
        print(counts)
        self.assertEqual(
            counts,
            [
                (
                    {
                        (("<root>", 0),): {"+": 1},
                        (("+", 0),): {"2": 1},
                        (("+", 1),): {"3": 1},
                        (("+", 2),): {"1": 1},
                    },
                    {
                        # note that there's no multiplicity here
                        # because the ordering mask doesn't allow it
                        (("<root>", 0),): {("+",): 1},
                        (("+", 0),): {("2",): 1},
                        (("+", 1),): {("3",): 1},
                        (("+", 2),): {("1",): 1},
                    },
                )
            ],
        )


class BigramParameterDifferenceLossTest(unittest.TestCase):
    def computeLoss(self, logits, programs, family=fam):
        programs = [[ns.parse_s_expression(x) for x in ps] for ps in programs]
        print(programs)
        loss = family.parameter_difference_loss(
            logits,
            family.count_programs(programs),
        )
        loss = loss.detach().cpu().numpy()
        return loss

    def assertLoss(self, logits, programs, target, **kwargs):
        np.testing.assert_almost_equal(
            self.computeLoss(logits, programs, **kwargs), target
        )

    def assertUniformLoss(self, programs, target, **kwargs):
        logits = torch.zeros((1, 4, 2, 4))
        self.assertLoss(logits, programs, target, **kwargs)
        # fill in impossible location +, 0 -> <root>
        logits[0, 1, 0, 0] = 20
        self.assertLoss(logits, programs, target, **kwargs)

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

    def test_variables_loss(self):
        logits = torch.zeros((1, 10, 2, 10))
        self.assertLoss(logits, [["(1)"]], [np.log(4)], family=fam_with_vars)
        self.assertLoss(
            logits,
            [["(call (lam ($0_0)) (1))"]],
            [np.log(4 * 5 * 4)],
            family=fam_with_vars,
        )


class BigramLikelihoodTest(unittest.TestCase):
    def assertLikelihood(self, dist, program, str_prob, family=fam, **kwargs):
        likelihood = family.compute_likelihood(
            dist, ns.parse_s_expression(program), **kwargs
        )
        self.assertEqual(self.render_likelihood(likelihood), str_prob)

    def assertLikelihoods(self, dist, program, nodes_and_probs, family=fam):
        likelihoods = family.compute_likelihood_per_node(
            dist, ns.parse_s_expression(program)
        )
        likelihoods = [
            (ns.render_s_expression(node), self.render_likelihood(prob))
            for node, prob in likelihoods
        ]
        print(likelihoods)
        self.assertEqual(likelihoods, nodes_and_probs)

    def render_likelihood(self, likelihood):
        prob = np.exp(likelihood)
        prob = Fraction.from_float(float(prob)).limit_denominator(1000)
        result = f"log({prob})"
        return result

    def test_leaf(self):
        self.assertLikelihood(fam.uniform(), "(1)", "log(1/3)")

    def test_plus(self):
        self.assertLikelihood(fam.uniform(), "(+ (1) (2))", "log(1/27)")

    def test_plus_nested(self):
        self.assertLikelihood(fam.uniform(), "(+ (1) (+ (1) (2)))", "log(1/243)")
        self.assertLikelihoods(
            fam.uniform(),
            "(+ (1) (+ (1) (2)))",
            [
                ("(+ (1) (+ (1) (2)))", "log(1/3)"),
                ("(1)", "log(1/3)"),
                ("(+ (1) (2))", "log(1/3)"),
                ("(1)", "log(1/3)"),
                ("(2)", "log(1/3)"),
            ],
        )

    def test_leaf_with_variables(self):
        self.assertLikelihood(
            fam_with_vars.uniform(), "(1)", "log(1/4)", family=fam_with_vars
        )

    def test_call_with_variables(self):
        # 1/4 for call
        # 1 for lam
        # 1/5 for $0_0
        # 1/4 for 1
        self.assertLikelihood(
            fam_with_vars.uniform(),
            "(call (lam ($0_0)) (1))",
            "log(1/80)",
            family=fam_with_vars,
        )
        self.assertLikelihoods(
            fam_with_vars.uniform(),
            "(call (lam ($0_0)) (1))",
            [
                ("(call (lam ($0_0)) (1))", "log(1/4)"),
                ("(lam ($0_0))", "log(1)"),
                ("($0_0)", "log(1/5)"),
                ("(1)", "log(1/4)"),
            ],
            family=fam_with_vars,
        )

    def test_ordered(self):
        self.assertLikelihood(
            fam_with_ordering.uniform(),
            "(+ (1) (2) (3))",
            "log(1)",
            family=fam_with_ordering,
        )
        self.assertLikelihood(
            fam_with_ordering.uniform(),
            "(+ (2) (3) (1))",
            "log(0)",
            family=fam_with_ordering,
        )

    def test_ordered_tracker(self):
        def tracker(node, likelihood):
            tracked.append(
                (ns.render_s_expression(node), self.render_likelihood(likelihood))
            )

        tracked = []
        self.assertLikelihood(
            fam_with_ordering.uniform(),
            "(+ (1) (2) (3))",
            "log(1)",
            family=fam_with_ordering,
            tracker=tracker,
        )
        print(tracked)
        self.assertEqual(
            tracked,
            [
                ("(+ (1) (2) (3))", "log(1)"),
                ("(1)", "log(1)"),
                ("(2)", "log(1)"),
                ("(3)", "log(1)"),
            ],
        )

        tracked = []
        self.assertLikelihood(
            fam_with_ordering.uniform(),
            "(+ (2) (3) (1))",
            "log(0)",
            family=fam_with_ordering,
            tracker=tracker,
        )
        print(tracked)
        self.assertEqual(
            tracked,
            [
                ("(+ (2) (3) (1))", "log(1)"),
                ("(2)", "log(0)"),
                ("(3)", "log(0)"),
                ("(1)", "log(0)"),
            ],
        )

    def test_ordered_asserting(self):
        self.assertLikelihood(
            fam_with_ordering_asserted.uniform(),
            "(+ (1) (2) (3))",
            "log(1/27)",
            family=fam_with_ordering_asserted,
        )
        # distribution where 1 is not allowed
        dist = fam_with_ordering_asserted.uniform()
        dist.distribution[:, :, 2] = 0
        self.assertLikelihood(
            dist,
            "(+ (1) (2) (3))",
            "log(0)",
            family=fam_with_ordering_asserted,
        )

    def test_ordered_231(self):
        self.assertLikelihood(
            fam_with_ordering_231.uniform(),
            "(+ (2) (3) (1))",
            "log(1)",
            family=fam_with_ordering_231,
        )
        self.assertLikelihood(
            fam_with_ordering_231.uniform(),
            "(+ (1) (2) (3))",
            "log(0)",
            family=fam_with_ordering_231,
        )

    def test_likelihood_clamped(self, kwargs=lambda x: {}):
        dist = fam.counts_to_distribution(
            fam.count_programs([[ns.parse_s_expression("(1)")]]),
        )[0]
        self.assertEqual(
            fam.compute_likelihood(dist, ns.parse_s_expression("(2)")), -np.inf
        )
        dist = dist.bound_minimum_likelihood(0.01, **kwargs(dist))
        # should be *very* approximately 1/100
        self.assertAlmostEqual(
            fam.compute_likelihood(dist, ns.parse_s_expression("(2)")),
            np.log(1 / 100),
            1,
        )
        # should be *very* approximately 1/100 * 1/3 * 1/3
        # even the (+ (1) (1)) should be since the + -> 1 bigram wasn't seen
        # in the original
        # the 1/3 comes from the fact that in the original the + -> 1 and + -> 2 bigrams
        # were undefined, so now they're uniform
        for code in "(+ (1) (2))", "(+ (1) (1))", "(+ (2) (2))":
            self.assertAlmostEqual(
                fam.compute_likelihood(dist, ns.parse_s_expression(code)),
                np.log(1 / 100 * 1 / 3 * 1 / 3),
                1,
            )

    def test_likelihood_clamped_mask_full(self):
        self.test_likelihood_clamped(
            lambda dist: dict(
                symbol_mask=np.ones((dist.distribution.shape[0]), dtype=np.bool_)
            )
        )

    @parameterized.expand([(True,), (False,)])
    def test_likelihood_clamped_mask_partial(self, include_type_mask):
        family = fam_with_3 if include_type_mask else fam_with_3_no_type_mask
        dist = family.counts_to_distribution(
            family.count_programs([[ns.parse_s_expression("(1)")]]),
        )[0]
        self.assertEqual(
            family.compute_likelihood(dist, ns.parse_s_expression("(2)")), -np.inf
        )
        self.assertEqual(
            [x for x, _ in family.tree_distribution_skeleton.symbols],
            ["<root>", "+", "1", "2", "3"],
        )
        print(np.array(np.where(dist.distribution)).T)
        dist = dist.bound_minimum_likelihood(
            0.01, symbol_mask=np.array([True, True, True, True, False])
        )
        print(np.array(np.where(dist.distribution)).T)
        # should be *very* approximately 1/100
        self.assertAlmostEqual(
            family.compute_likelihood(dist, ns.parse_s_expression("(2)")),
            np.log(1 / 100),
            1,
        )
        # should be *very* approximately 1/100
        self.assertAlmostEqual(
            family.compute_likelihood(dist, ns.parse_s_expression("(3)")),
            -np.inf,
            1,
        )

        # see test_likelihood_clamped for the math
        for code in "(+ (1) (2))", "(+ (1) (1))", "(+ (2) (2))":
            self.assertAlmostEqual(
                family.compute_likelihood(dist, ns.parse_s_expression(code)),
                np.log(1 / 100 * 1 / 3 * 1 / 3),
                1,
            )

        # remains -inf because 3 is impossible
        for code in "(+ (1) (3))", "(+ (3) (3))", "(+ (3) (2))":
            self.assertAlmostEqual(
                family.compute_likelihood(dist, ns.parse_s_expression(code)),
                -np.inf,
                1,
            )


class BigramEnumerationTest(unittest.TestCase):
    def test_enumeration_basic(self):
        dist = fam.uniform()

        self.assertEqual(
            enumerate_dsl(fam, dist, -5),
            {
                ("(1)", Fraction(1, 3)),
                ("(2)", Fraction(1, 3)),
                ("(+ (1) (1))", Fraction(1, 27)),
                ("(+ (1) (2))", Fraction(1, 27)),
                ("(+ (2) (1))", Fraction(1, 27)),
                ("(+ (2) (2))", Fraction(1, 27)),
            },
        )

    def test_enumeration_with_vars(self):
        dist = fam_with_vars.uniform()
        self.assertEqual(
            enumerate_dsl(fam_with_vars, dist, -5),
            {
                ("(1)", Fraction(1, 4)),
                ("(2)", Fraction(1, 4)),
                ("(+ (1) (1))", Fraction(1, 64)),
                ("(+ (1) (2))", Fraction(1, 64)),
                ("(+ (2) (1))", Fraction(1, 64)),
                ("(+ (2) (2))", Fraction(1, 64)),
                ("(call (lam ($0_0)) (1))", Fraction(1, 80)),
                ("(call (lam ($0_0)) (2))", Fraction(1, 80)),
                ("(call (lam (1)) (1))", Fraction(1, 80)),
                ("(call (lam (1)) (2))", Fraction(1, 80)),
                ("(call (lam (2)) (1))", Fraction(1, 80)),
                ("(call (lam (2)) (2))", Fraction(1, 80)),
            },
        )

    def test_enumeration_more_complex(self):
        dist = fam_more_complex.uniform()
        self.assertEqual(
            enumerate_dsl(fam_more_complex, dist, -9),
            {
                ("(lam ($0_0))", Fraction(1, 4)),
                ("(lam (sqrt (0)))", Fraction(1, 8)),
                ("(lam (sqrt (+ (0) (0))))", Fraction(1, 32)),
                ("(lam (* ($0_0) ($0_0)))", Fraction(1, 64)),
                ("(lam (* ($0_0) (sqrt (0))))", Fraction(1, 128)),
                ("(lam (* (sqrt (0)) ($0_0)))", Fraction(1, 128)),
                ("(lam (sqrt (+ (+ (0) (0)) (0))))", Fraction(1, 128)),
                ("(lam (sqrt (+ (0) (+ (0) (0)))))", Fraction(1, 128)),
                ("(lam (* (sqrt (0)) (sqrt (0))))", Fraction(1, 256)),
                ("(lam (* ($0_0) (sqrt (+ (0) (0)))))", Fraction(1, 512)),
                ("(lam (* (sqrt (+ (0) (0))) ($0_0)))", Fraction(1, 512)),
                ("(lam (ite (< (0) (0)) ($0_0) ($0_0)))", Fraction(1, 512)),
                ("(lam (sqrt (+ (+ (+ (0) (0)) (0)) (0))))", Fraction(1, 512)),
                ("(lam (sqrt (+ (+ (0) (+ (0) (0))) (0))))", Fraction(1, 512)),
                ("(lam (sqrt (+ (+ (0) (0)) (+ (0) (0)))))", Fraction(1, 512)),
                ("(lam (sqrt (+ (0) (+ (+ (0) (0)) (0)))))", Fraction(1, 512)),
                ("(lam (sqrt (+ (0) (+ (0) (+ (0) (0))))))", Fraction(1, 512)),
                ("(lam (* ($0_0) (* ($0_0) ($0_0))))", Fraction(1, 1024)),
                ("(lam (* (* ($0_0) ($0_0)) ($0_0)))", Fraction(1, 1024)),
                ("(lam (* (sqrt (+ (0) (0))) (sqrt (0))))", Fraction(1, 1024)),
                ("(lam (* (sqrt (0)) (sqrt (+ (0) (0)))))", Fraction(1, 1024)),
                ("(lam (ite (< (0) (0)) ($0_0) (sqrt (0))))", Fraction(1, 1024)),
                ("(lam (ite (< (0) (0)) (sqrt (0)) ($0_0)))", Fraction(1, 1024)),
                ("(lam (* ($0_0) (* ($0_0) (sqrt (0)))))", Fraction(1, 2048)),
                ("(lam (* ($0_0) (* (sqrt (0)) ($0_0))))", Fraction(1, 2048)),
                ("(lam (* ($0_0) (sqrt (+ (+ (0) (0)) (0)))))", Fraction(1, 2048)),
                ("(lam (* ($0_0) (sqrt (+ (0) (+ (0) (0))))))", Fraction(1, 2048)),
                ("(lam (* (* ($0_0) ($0_0)) (sqrt (0))))", Fraction(1, 2048)),
                ("(lam (* (* ($0_0) (sqrt (0))) ($0_0)))", Fraction(1, 2048)),
                ("(lam (* (* (sqrt (0)) ($0_0)) ($0_0)))", Fraction(1, 2048)),
                ("(lam (* (sqrt (+ (+ (0) (0)) (0))) ($0_0)))", Fraction(1, 2048)),
                ("(lam (* (sqrt (+ (0) (+ (0) (0)))) ($0_0)))", Fraction(1, 2048)),
                ("(lam (* (sqrt (0)) (* ($0_0) ($0_0))))", Fraction(1, 2048)),
                ("(lam (ite (< (+ (0) (0)) (0)) ($0_0) ($0_0)))", Fraction(1, 2048)),
                ("(lam (ite (< (0) (+ (0) (0))) ($0_0) ($0_0)))", Fraction(1, 2048)),
                ("(lam (ite (< (0) (0)) (sqrt (0)) (sqrt (0))))", Fraction(1, 2048)),
                ("(lam (ite (> ($0_0) ($0_0)) ($0_0) ($0_0)))", Fraction(1, 2048)),
                ("(lam (sqrt (+ (+ (+ (+ (0) (0)) (0)) (0)) (0))))", Fraction(1, 2048)),
                ("(lam (sqrt (+ (+ (+ (0) (+ (0) (0))) (0)) (0))))", Fraction(1, 2048)),
                ("(lam (sqrt (+ (+ (+ (0) (0)) (+ (0) (0))) (0))))", Fraction(1, 2048)),
                ("(lam (sqrt (+ (+ (+ (0) (0)) (0)) (+ (0) (0)))))", Fraction(1, 2048)),
                ("(lam (sqrt (+ (+ (0) (+ (+ (0) (0)) (0))) (0))))", Fraction(1, 2048)),
                ("(lam (sqrt (+ (+ (0) (+ (0) (+ (0) (0)))) (0))))", Fraction(1, 2048)),
                ("(lam (sqrt (+ (+ (0) (+ (0) (0))) (+ (0) (0)))))", Fraction(1, 2048)),
                ("(lam (sqrt (+ (+ (0) (0)) (+ (+ (0) (0)) (0)))))", Fraction(1, 2048)),
                ("(lam (sqrt (+ (+ (0) (0)) (+ (0) (+ (0) (0))))))", Fraction(1, 2048)),
                ("(lam (sqrt (+ (0) (+ (+ (+ (0) (0)) (0)) (0)))))", Fraction(1, 2048)),
                ("(lam (sqrt (+ (0) (+ (+ (0) (+ (0) (0))) (0)))))", Fraction(1, 2048)),
                ("(lam (sqrt (+ (0) (+ (+ (0) (0)) (+ (0) (0))))))", Fraction(1, 2048)),
                ("(lam (sqrt (+ (0) (+ (0) (+ (+ (0) (0)) (0))))))", Fraction(1, 2048)),
                ("(lam (sqrt (+ (0) (+ (0) (+ (0) (+ (0) (0)))))))", Fraction(1, 2048)),
                ("(lam (* ($0_0) (* (sqrt (0)) (sqrt (0)))))", Fraction(1, 4096)),
                ("(lam (* (* ($0_0) (sqrt (0))) (sqrt (0))))", Fraction(1, 4096)),
                ("(lam (* (* (sqrt (0)) ($0_0)) (sqrt (0))))", Fraction(1, 4096)),
                ("(lam (* (* (sqrt (0)) (sqrt (0))) ($0_0)))", Fraction(1, 4096)),
                ("(lam (* (sqrt (+ (+ (0) (0)) (0))) (sqrt (0))))", Fraction(1, 4096)),
                ("(lam (* (sqrt (+ (0) (+ (0) (0)))) (sqrt (0))))", Fraction(1, 4096)),
                ("(lam (* (sqrt (+ (0) (0))) (sqrt (+ (0) (0)))))", Fraction(1, 4096)),
                ("(lam (* (sqrt (0)) (* ($0_0) (sqrt (0)))))", Fraction(1, 4096)),
                ("(lam (* (sqrt (0)) (* (sqrt (0)) ($0_0))))", Fraction(1, 4096)),
                ("(lam (* (sqrt (0)) (sqrt (+ (+ (0) (0)) (0)))))", Fraction(1, 4096)),
                ("(lam (* (sqrt (0)) (sqrt (+ (0) (+ (0) (0))))))", Fraction(1, 4096)),
                (
                    "(lam (ite (< (+ (0) (0)) (0)) ($0_0) (sqrt (0))))",
                    Fraction(1, 4096),
                ),
                (
                    "(lam (ite (< (+ (0) (0)) (0)) (sqrt (0)) ($0_0)))",
                    Fraction(1, 4096),
                ),
                (
                    "(lam (ite (< (0) (+ (0) (0))) ($0_0) (sqrt (0))))",
                    Fraction(1, 4096),
                ),
                (
                    "(lam (ite (< (0) (+ (0) (0))) (sqrt (0)) ($0_0)))",
                    Fraction(1, 4096),
                ),
                (
                    "(lam (ite (< (0) (0)) ($0_0) (sqrt (+ (0) (0)))))",
                    Fraction(1, 4096),
                ),
                (
                    "(lam (ite (< (0) (0)) (sqrt (+ (0) (0))) ($0_0)))",
                    Fraction(1, 4096),
                ),
                ("(lam (ite (> ($0_0) ($0_0)) ($0_0) (sqrt (0))))", Fraction(1, 4096)),
                ("(lam (ite (> ($0_0) ($0_0)) (sqrt (0)) ($0_0)))", Fraction(1, 4096)),
                ("(lam (ite (> ($0_0) (sqrt (0))) ($0_0) ($0_0)))", Fraction(1, 4096)),
                ("(lam (ite (> (sqrt (0)) ($0_0)) ($0_0) ($0_0)))", Fraction(1, 4096)),
            },
        )

    def test_enumeration_more_complex_get_many(self):
        dist = fam_more_complex.uniform()
        large_set = sorted(
            enumerate_dsl(fam_more_complex, dist, -12), key=lambda x: (-x[1], x[0])
        )
        self.assertEqual(len(large_set), 1609)
        print(large_set[::100])
        self.assertEqual(
            large_set[::100],
            [
                ("(lam ($0_0))", Fraction(1, 4)),
                ("(lam (ite (< (0) (0)) (* ($0_0) ($0_0)) ($0_0)))", Fraction(1, 8192)),
                (
                    "(lam (ite (< (0) (+ (0) (0))) ($0_0) (sqrt (+ (0) (0)))))",
                    Fraction(1, 16384),
                ),
                (
                    "(lam (ite (< (+ (+ (0) (0)) (+ (0) (0))) (0)) ($0_0) ($0_0)))",
                    Fraction(1, 32768),
                ),
                (
                    "(lam (sqrt (+ (+ (+ (0) (0)) (0)) (+ (0) (+ (0) (+ (0) (0)))))))",
                    Fraction(1, 32768),
                ),
                (
                    "(lam (* ($0_0) (ite (< (0) (0)) ($0_0) (sqrt (+ (0) (0))))))",
                    Fraction(1, 65536),
                ),
                (
                    "(lam (* (sqrt (0)) (ite (< (0) (+ (0) (0))) ($0_0) ($0_0))))",
                    Fraction(1, 65536),
                ),
                (
                    "(lam (ite (> ($0_0) ($0_0)) (* ($0_0) (sqrt (0))) ($0_0)))",
                    Fraction(1, 65536),
                ),
                (
                    "(lam (* ($0_0) (sqrt (+ (0) (+ (+ (+ (+ (0) (0)) (0)) (0)) (0))))))",
                    Fraction(1, 131072),
                ),
                (
                    "(lam (* (sqrt (+ (+ (+ (0) (+ (+ (0) (0)) (0))) (0)) (0))) ($0_0)))",
                    Fraction(1, 131072),
                ),
                (
                    "(lam (ite (< (+ (0) (+ (0) (+ (0) (0)))) (+ (0) (0))) ($0_0) ($0_0)))",
                    Fraction(1, 131072),
                ),
                (
                    "(lam (ite (< (0) (0)) (sqrt (0)) (ite (< (0) (0)) ($0_0) ($0_0))))",
                    Fraction(1, 131072),
                ),
                (
                    "(lam (sqrt (+ (+ (+ (+ (+ (0) (0)) (0)) (+ (0) (0))) (+ (0) (0))) (0))))",
                    Fraction(1, 131072),
                ),
                (
                    "(lam (sqrt (+ (+ (+ (0) (+ (0) (0))) (0)) (+ (+ (0) (0)) (+ (0) (0))))))",
                    Fraction(1, 131072),
                ),
                (
                    "(lam (sqrt (+ (+ (0) (+ (0) (+ (+ (0) (+ (0) (0))) (+ (0) (0))))) (0))))",
                    Fraction(1, 131072),
                ),
                (
                    "(lam (sqrt (+ (0) (+ (+ (+ (0) (+ (+ (0) (0)) (0))) (+ (0) (0))) (0)))))",
                    Fraction(1, 131072),
                ),
                (
                    "(lam (sqrt (+ (0) (+ (0) (+ (0) (+ (+ (0) (+ (0) (+ (0) (0)))) (0)))))))",
                    Fraction(1, 131072),
                ),
            ],
        )

    def test_enumeration_more_complex_get_even_more(self):
        dist = fam_more_complex.uniform()
        large_set = sorted(
            enumerate_dsl(fam_more_complex, dist, -15, max_denominator=10**8),
            key=lambda x: (-x[1], x[0]),
        )
        self.assertEqual(len(large_set), 18804)
        print(large_set[::1000])
        self.assertEqual(
            large_set[::1000],
            [
                ("(lam ($0_0))", Fraction(1, 4)),
                (
                    "(lam (ite (< (+ (0) (+ (0) (+ (0) (0)))) (+ (0) (0))) ($0_0) ($0_0)))",
                    Fraction(707, 92667907),
                ),
                (
                    "(lam (* (sqrt (0)) (sqrt (+ (+ (+ (+ (0) (0)) (0)) (0)) (+ (0) (0))))))",
                    Fraction(334, 87556099),
                ),
                (
                    "(lam (* (ite (> ($0_0) (sqrt (0))) (sqrt (+ (0) (0))) ($0_0)) ($0_0)))",
                    Fraction(158, 82837507),
                ),
                (
                    "(lam (sqrt (+ (+ (+ (+ (+ (+ (0) (+ (0) (+ (0) (0)))) (0)) (0)) (0)) (0)) (0))))",
                    Fraction(158, 82837507),
                ),
                (
                    "(lam (sqrt (+ (0) (+ (+ (+ (+ (+ (0) (+ (0) (+ (0) (0)))) (0)) (0)) (0)) (0)))))",
                    Fraction(158, 82837507),
                ),
                (
                    "(lam (* (* (sqrt (0)) ($0_0)) (sqrt (+ (0) (+ (+ (+ (0) (0)) (0)) (0))))))",
                    Fraction(25, 26214401),
                ),
                (
                    "(lam (ite (< (+ (+ (+ (0) (0)) (0)) (0)) (+ (0) (0))) (sqrt (+ (0) (0))) ($0_0)))",
                    Fraction(25, 26214401),
                ),
                (
                    "(lam (ite (> ($0_0) ($0_0)) (sqrt (+ (0) (+ (0) (+ (0) (+ (0) (0)))))) ($0_0)))",
                    Fraction(25, 26214401),
                ),
                (
                    "(lam (* ($0_0) (sqrt (+ (+ (+ (0) (0)) (0)) (+ (+ (0) (+ (0) (+ (0) (0)))) (0))))))",
                    Fraction(12, 25165825),
                ),
                (
                    "(lam (* (ite (< (+ (0) (+ (0) (0))) (0)) ($0_0) (sqrt (+ (0) (0)))) (sqrt (0))))",
                    Fraction(12, 25165825),
                ),
                (
                    "(lam (* (sqrt (+ (0) (0))) (ite (< (0) (0)) (* ($0_0) (sqrt (0))) ($0_0))))",
                    Fraction(12, 25165825),
                ),
                (
                    "(lam (ite (< (+ (0) (0)) (0)) (* ($0_0) (sqrt (0))) (* ($0_0) (sqrt (0)))))",
                    Fraction(12, 25165825),
                ),
                (
                    "(lam (ite (< (0) (0)) (sqrt (0)) (sqrt (+ (+ (0) (+ (0) (+ (+ (0) (0)) (0)))) (0)))))",
                    Fraction(12, 25165825),
                ),
                (
                    "(lam (sqrt (+ (+ (+ (+ (+ (+ (0) (+ (0) (+ (0) (0)))) (0)) (0)) (0)) (0)) (+ (0) (0)))))",
                    Fraction(12, 25165825),
                ),
                (
                    "(lam (sqrt (+ (+ (+ (0) (+ (+ (+ (0) (+ (0) (0))) (0)) (0))) (+ (+ (0) (0)) (0))) (0))))",
                    Fraction(12, 25165825),
                ),
                (
                    "(lam (sqrt (+ (+ (0) (+ (+ (+ (+ (0) (+ (0) (0))) (0)) (+ (0) (0))) (0))) (+ (0) (0)))))",
                    Fraction(12, 25165825),
                ),
                (
                    "(lam (sqrt (+ (+ (0) (0)) (+ (+ (+ (+ (0) (0)) (+ (0) (+ (0) (0)))) (0)) (+ (0) (0))))))",
                    Fraction(12, 25165825),
                ),
                (
                    "(lam (sqrt (+ (0) (+ (+ (0) (+ (+ (+ (0) (0)) (+ (0) (+ (0) (0)))) (+ (0) (0)))) (0)))))",
                    Fraction(12, 25165825),
                ),
            ],
        )

    def test_enumeration_more_complex_get_even_more_even_more(self):
        dist = fam_more_complex.uniform()
        large_set = sorted(
            enumerate_dsl(fam_more_complex, dist, -17, max_denominator=10**8),
            key=lambda x: (-x[1], x[0]),
        )
        self.assertEqual(len(large_set), 103_258)
        print(large_set[::10000])
        self.assertEqual(
            large_set[::10000],
            [
                ("(lam ($0_0))", Fraction(1, 4)),
                (
                    "(lam (* (ite (< (+ (0) (+ (0) (0))) (0)) ($0_0) (sqrt (+ (0) (0)))) (sqrt (0))))",
                    Fraction(12, 25165825),
                ),
                (
                    "(lam (* (* ($0_0) (sqrt (0))) (* (sqrt (+ (0) (+ (0) (+ (0) (0))))) ($0_0))))",
                    Fraction(17, 71303171),
                ),
                (
                    "(lam (* ($0_0) (* (sqrt (0)) (* ($0_0) (ite (< (0) (0)) (sqrt (0)) ($0_0))))))",
                    Fraction(11, 92274692),
                ),
                (
                    "(lam (ite (< (+ (+ (0) (+ (0) (+ (0) (0)))) (0)) (0)) (sqrt (+ (+ (0) (0)) (0))) (sqrt (0))))",
                    Fraction(11, 92274692),
                ),
                (
                    "(lam (sqrt (+ (+ (+ (+ (+ (0) (0)) (+ (+ (0) (+ (0) (0))) (0))) (+ (0) (+ (0) (0)))) (0)) (0))))",
                    Fraction(11, 92274692),
                ),
                (
                    "(lam (sqrt (+ (+ (0) (0)) (+ (+ (0) (+ (+ (+ (0) (+ (0) (0))) (0)) (+ (0) (0)))) (+ (0) (0))))))",
                    Fraction(11, 92274692),
                ),
                (
                    "(lam (* (* ($0_0) (sqrt (0))) (ite (< (+ (+ (0) (0)) (0)) (+ (0) (0))) ($0_0) ($0_0))))",
                    Fraction(4, 67108867),
                ),
                (
                    "(lam (* (sqrt (+ (0) (+ (0) (+ (+ (0) (+ (0) (+ (0) (0)))) (0))))) (sqrt (+ (0) (+ (0) (0))))))",
                    Fraction(4, 67108867),
                ),
                (
                    "(lam (ite (< (+ (0) (0)) (0)) ($0_0) (ite (> (sqrt (0)) ($0_0)) (sqrt (+ (0) (0))) ($0_0))))",
                    Fraction(4, 67108867),
                ),
                (
                    "(lam (ite (> (* (* ($0_0) ($0_0)) (sqrt (0))) ($0_0)) (sqrt (+ (0) (0))) (sqrt (0))))",
                    Fraction(4, 67108867),
                ),
            ],
        )


class BigramMixTest(unittest.TestCase):
    def fit_family_to_programs(self, family, programs):
        return family.counts_to_distribution(
            family.count_programs([[ns.parse_s_expression(x) for x in programs]]),
        )[0]

    def merged(self, fam_1, programs_1, fam_2, programs_2, weight):
        dist_1 = self.fit_family_to_programs(fam_1, programs_1)
        dist_2 = self.fit_family_to_programs(fam_2, programs_2)
        return dist_1.mix_with_other(dist_2, weight)

    def assertClose(self, dist_merged, dist_expected):
        delta = dist_merged.distribution - dist_expected.distribution
        bad = np.abs(delta) > 1e-6
        if bad.any():
            syms = dist_merged.dist_fam.symbols()
            for parent, which in np.array(np.where(bad.any(-1))).T:
                actual = dist_merged.distribution[parent, which]
                expected = dist_expected.distribution[parent, which]
                select = (actual > 0) | (expected > 0)
                actual, expected = [
                    {syms[i]: arr[i] for i in np.where(select)[0]}
                    for arr in (actual, expected)
                ]
                self.assertEqual(
                    actual, expected, f"At {syms[parent]}'s {which}th child"
                )

    def assertMix(self, fam_1, programs_1, fam_2, programs_2, weight_1, weight_2):
        self.assertClose(
            self.merged(
                fam_1, programs_1, fam_2, programs_2, weight_2 / (weight_1 + weight_2)
            ),
            self.fit_family_to_programs(
                fam_1 if len(fam_1.symbols()) > len(fam_2.symbols()) else fam_2,
                programs_1 * weight_1 + programs_2 * weight_2,
            ),
        )

    def assertMixBoth(self, fam_1, programs_1, fam_2, programs_2, weight_1, weight_2):
        # pylint: disable=arguments-out-of-order
        self.assertMix(fam_1, programs_1, fam_2, programs_2, weight_1, weight_2)
        self.assertMix(fam_2, programs_2, fam_1, programs_1, weight_2, weight_1)

    progs_1 = "(1)", "(+ (1) (2))"
    progs_2 = "(+ (2) (2))", "(2)"
    progs_3 = "(+ (2) (3))", "(3)"

    def test_merge_bigrams_same_dsl_equal_weight(self):
        self.assertMixBoth(fam, self.progs_1, fam, self.progs_2, 1, 1)

    def test_merge_bigrams_same_dsl_more_weight(self):
        self.assertMixBoth(fam, self.progs_1, fam, self.progs_2, 2, 1)

    def test_merge_bigrams_same_dsl_less_weight(self):
        self.assertMixBoth(fam, self.progs_1, fam, self.progs_2, 2, 5)

    def test_merge_bigrams_with_different_fams_equal_weight(self):
        self.assertMixBoth(fam, self.progs_1, fam_with_3, self.progs_3, 1, 1)

    def test_merge_bigrams_with_different_fams_more_weight(self):
        self.assertMixBoth(fam, self.progs_1, fam_with_3, self.progs_3, 9, 4)

    def test_merge_bigrams_with_different_fams_less_weight(self):
        self.assertMixBoth(fam, self.progs_1, fam_with_3, self.progs_3, 2, 4)

    def test_non_subset_fails(self):
        with self.assertRaises(ValueError) as cm:
            self.merged(fam_with_3, self.progs_3, fam_with_vars, self.progs_1, 1)

        self.assertEqual(
            str(cm.exception),
            "DSL not compatible, extra symbols in this: '3', "
            "extra symbols in other: '$0_0', '$1_0', '$2_0', '$3_0', 'call', 'lam'",
        )


class OrderingTest(unittest.TestCase):
    def test_traversal(self):
        ordering = fam_with_ordering.tree_distribution_skeleton.ordering
        self.assertEqual(
            [
                ns.render_s_expression(x)
                for x in ordering.traverse_preorder(
                    ns.parse_s_expression("(+ (1) (2) (3))")
                )
            ],
            ["(+ (1) (2) (3))", "(1)", "(2)", "(3)"],
        )

    def test_traversal_231(self):
        ordering = fam_with_ordering_231.tree_distribution_skeleton.ordering
        self.assertEqual(
            [
                ns.render_s_expression(x)
                for x in ordering.traverse_preorder(
                    ns.parse_s_expression("(+ (2) (3) (1))")
                )
            ],
            ["(+ (2) (3) (1))", "(1)", "(2)", "(3)"],
        )
        self.assertEqual(
            [
                ns.render_s_expression(x)
                for x in ordering.traverse_preorder(
                    ns.parse_s_expression("(+ (1) (2) (3))")
                )
            ],
            ["(+ (1) (2) (3))", "(3)", "(1)", "(2)"],
        )
