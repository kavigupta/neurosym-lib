import unittest

import numpy as np

import neurosym as ns


class DSLSmoothingTest(unittest.TestCase):
    def setUp(self):
        self.dfa = None
        self.fam = None
        self.dist = None

    def dist_for_smoothing(
        self, programs, extra_programs=(), root="M", do_smooth_masking=False
    ):
        programs = [ns.python_to_python_ast(program) for program in programs]
        extra_programs = [
            ns.python_to_python_ast(program) for program in extra_programs
        ]
        dfa = ns.python_dfa()
        dsl = ns.create_python_dsl(
            dfa,
            ns.PythonDSLSubset.from_programs(
                dfa, *programs, *extra_programs, root=root
            ),
            root,
        )
        if do_smooth_masking:
            dsl_subset = ns.create_python_dsl(
                dfa,
                ns.PythonDSLSubset.from_programs(dfa, *programs, root=root),
                root,
            )
            smooth_mask = dsl.create_smoothing_mask(dsl_subset)
        else:
            smooth_mask = None
        fam = ns.BigramProgramDistributionFamily(
            dsl,
            additional_preorder_masks=[
                lambda dist, dsl: ns.python_def_use_mask.DefUseChainPreorderMask(
                    dist, dsl, ns.python_def_use_mask.DefUseMaskConfiguration(dfa, {})
                )
            ],
            node_ordering=ns.python_def_use_mask.PythonNodeOrdering,
        )
        counts = fam.count_programs(
            [
                [
                    ns.to_type_annotated_ns_s_exp(program, dfa, root)
                    for program in programs
                ]
            ]
        )
        dist = fam.counts_to_distribution(counts)[0]
        dist = dist.bound_minimum_likelihood(1e-6, smooth_mask)
        self.dfa = dfa
        self.fam = fam
        self.dist = dist

    def likelihood(self, program):
        program = ns.python_to_python_ast(program)
        log_p = self.fam.compute_likelihood(
            self.dist,
            ns.to_type_annotated_ns_s_exp(program, self.dfa, "M"),
        )
        return log_p

    def assertLikeli(self, prog, reciprocal_prob):
        self.assertAlmostEqual(self.likelihood(prog), -np.log(reciprocal_prob), 3)

    def standard_checks(self):
        # 1/2 [2], 1/2 [5]
        self.assertLikeli("x = 2 + 5", 2 * 2)
        # 1/2 [2], 1/1e6 [+], 1/2 [5], 1/2 [5]
        self.assertLikeli("x = 2 + (2 + 5)", 2 * 10**6 * 2 * 2)

    def test_smoothing_no_extra(self):
        self.dist_for_smoothing(["x = 2 + 5"])
        self.standard_checks()

    def test_smoothing_with_extra(self):
        self.dist_for_smoothing(["x = 2 + 5"], ["x = 2 * 3"])
        self.standard_checks()
        # 1/1e6 [*], 1/2 [2], 1/2 [5]
        self.assertLikeli("x = 2 * 5", 10**6 * 2 * 2)

    def test_smoothing_with_extra_subset_mask(self):
        self.dist_for_smoothing(["x = 2 + 5"], ["x = 2 * 3"], do_smooth_masking=True)
        self.standard_checks()
        # 1/1e6 [*], 1/2 [2], 1/2 [5]
        self.assertLikeli("x = 2 * 5", np.inf)
