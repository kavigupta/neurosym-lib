import unittest

import neurosym as ns


class TestTowerDSL(unittest.TestCase):

    def assertProgramRun(self, program, expected_end_state, expected_plan):
        dsl = ns.examples.tower.tower_dsl()
        program = ns.parse_s_expression(program)
        end_state, plan = dsl.compute(dsl.initialize(program))(
            ns.examples.tower.TowerState()
        )
        self.assertEqual(plan, expected_plan)
        self.assertEqual(str(end_state), expected_end_state)

    def test_basic_program(self):
        self.assertProgramRun("(semi (r (4)) (v))", "S(h=4,o=1)", [(4, 2, 6)])

    def test_basic_repeat_loop(self):
        self.assertProgramRun(
            "(for (4) (lam (semi (r (4)) (v))))",
            "S(h=16,o=1)",
            [(4, 2, 6), (8, 2, 6), (12, 2, 6), (16, 2, 6)],
        )

    def test_basic_for_loop(self):
        self.assertProgramRun(
            "(for (4) (lam (semi (r ($0_0)) (v))))",
            "S(h=6,o=1)",
            [(0, 2, 6), (1, 2, 6), (3, 2, 6), (6, 2, 6)],
        )

    def test_nested_for_loop(self):
        self.assertProgramRun(
            "(for (4) (lam (for (4) (lam (semi (r ($0_0)) (semi (v) (semi (r ($1_0)) (h))))))))",
            "S(h=48,o=1)",
            [
                (0, 2, 6),
                (0, 6, 2),
                (1, 2, 6),
                (1, 6, 2),
                (3, 2, 6),
                (3, 6, 2),
                (6, 2, 6),
                (6, 6, 2),
                (6, 2, 6),
                (7, 6, 2),
                (8, 2, 6),
                (9, 6, 2),
                (11, 2, 6),
                (12, 6, 2),
                (15, 2, 6),
                (16, 6, 2),
                (16, 2, 6),
                (18, 6, 2),
                (19, 2, 6),
                (21, 6, 2),
                (23, 2, 6),
                (25, 6, 2),
                (28, 2, 6),
                (30, 6, 2),
                (30, 2, 6),
                (33, 6, 2),
                (34, 2, 6),
                (37, 6, 2),
                (39, 2, 6),
                (42, 6, 2),
                (45, 2, 6),
                (48, 6, 2),
            ],
        )
