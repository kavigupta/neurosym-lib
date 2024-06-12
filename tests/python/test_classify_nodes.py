import unittest

import neurosym as ns


class TestBasicAnnotation(unittest.TestCase):
    def assertBasicAnnotation(self, dfa, program, state, expected):
        result = ns.run_dfa_on_program(dfa, ns.parse_s_expression(program), state)
        self.assertEqual(
            [(ns.render_s_expression(node), state) for node, state in result], expected
        )

    def assertDisambiguatingTypeTags(self, dfa, prog, start_state, expected):
        result = ns.add_disambiguating_type_tags(
            dfa, ns.parse_s_expression(prog), start_state
        )
        self.assertEqual(ns.render_s_expression(result), expected)

    def test_basic(self):
        dfa = {
            "start": {
                "a": ["state1", "state2"],
            },
            "state1": {
                "b": ["state2"],
            },
            "state2": {
                "c": [],
            },
        }
        self.assertBasicAnnotation(
            dfa,
            "(a (b (c)) (c))",
            "start",
            [
                ("(a (b (c)) (c))", "start"),
                ("(b (c))", "state1"),
                ("(c)", "state2"),
                ("(c)", "state2"),
            ],
        )
        self.assertDisambiguatingTypeTags(
            dfa,
            "(a (b (c)) (c))",
            "start",
            "(a~start (b~state1 (c~state2)) (c~state2))",
        )

    def test_repeated_state(self):
        dfa = {
            "start": {
                "a": ["[state1]", "state1"],
            },
            "[state1]": {
                "list": ["state1"],
            },
            "state1": {
                "b": [],
            },
        }
        self.assertBasicAnnotation(
            dfa,
            "(a (list (b) (b) (b)))",
            "start",
            [
                ("(a (list (b) (b) (b)))", "start"),
                ("(list (b) (b) (b))", "[state1]"),
                ("(b)", "state1"),
                ("(b)", "state1"),
                ("(b)", "state1"),
            ],
        )
        self.assertDisambiguatingTypeTags(
            dfa,
            "(a (list (b) (b) (b)))",
            "start",
            "(a~start (list~_state1_~3 (b~state1) (b~state1) (b~state1)))",
        )
