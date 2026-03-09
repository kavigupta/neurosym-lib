import unittest

import torch

import neurosym as ns
from neurosym.examples import near


class TestSimpleECGDSL(unittest.TestCase):
    def setUp(self):
        self.input_dim = 12 * 6 * 2
        self.num_classes = 5
        self.dsl = near.simple_ecg_dsl(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
        )

    def test_has_group_and_drop_productions(self):
        symbols = {production.symbol() for production in self.dsl.productions}
        self.assertIn("drop_variables", symbols)
        self.assertIn("channel_group_limb", symbols)
        self.assertIn("channel_group_precordial", symbols)

    def test_group_selector_averages_selected_channels(self):
        program = ns.SExpression(
            symbol="select_interval",
            children=(ns.SExpression(symbol="channel_group_all", children=()),),
        )
        selector = self.dsl.compute(self.dsl.initialize(program))

        x = torch.arange(self.input_dim, dtype=torch.float32).reshape(1, self.input_dim)
        selected = selector(x)
        expected = x.reshape(1, 12, 6, 2).mean(dim=1)[:, :, 0]
        self.assertTrue(torch.allclose(selected, expected))

    def test_drop_variables_can_remove_all_channels(self):
        program = ns.SExpression(
            symbol="select_amplitude",
            children=(
                ns.SExpression(
                    symbol="drop_variables",
                    children=(
                        ns.SExpression(symbol="channel_group_all", children=()),
                        ns.SExpression(symbol="channel_group_all", children=()),
                    ),
                ),
            ),
        )
        selector = self.dsl.compute(self.dsl.initialize(program))
        x = torch.randn(3, self.input_dim)
        selected = selector(x)
        self.assertTrue(torch.allclose(selected, torch.zeros_like(selected)))

    def test_drop_variables_program_executes(self):
        program = ns.SExpression(
            symbol="output",
            children=(
                ns.SExpression(
                    symbol="select_amplitude",
                    children=(
                        ns.SExpression(
                            symbol="drop_variables",
                            children=(
                                ns.SExpression(
                                    symbol="channel_group_all",
                                    children=(),
                                ),
                                ns.SExpression(
                                    symbol="channel_group_precordial",
                                    children=(),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        model = self.dsl.compute(self.dsl.initialize(program))
        output = model(torch.randn(4, self.input_dim))
        self.assertEqual(tuple(output.shape), (4, self.num_classes))
