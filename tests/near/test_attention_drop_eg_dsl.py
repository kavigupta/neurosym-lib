import unittest

import torch

import neurosym as ns
from neurosym.examples import near


class TestAttentionDropEGDSL(unittest.TestCase):
    def setUp(self):
        self.input_dim = 12 * 6 * 2
        self.num_classes = 4
        self.dsl = near.attention_drop_eg_dsl(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
        )

    def test_productions_exist(self):
        symbols = {production.symbol() for production in self.dsl.productions}
        self.assertIn("drop_variables", symbols)
        self.assertIn("attention_interval", symbols)
        self.assertIn("channel_group_precordial", symbols)

    def test_program_executes(self):
        program = ns.SExpression(
            symbol="output",
            children=(
                ns.SExpression(
                    symbol="attention_interval",
                    children=(
                        ns.SExpression(
                            symbol="drop_variables",
                            children=(
                                ns.SExpression(symbol="channel_group_all", children=()),
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
        out = model(torch.randn(3, self.input_dim))
        self.assertEqual(tuple(out.shape), (3, self.num_classes))
        self.assertTrue(torch.isfinite(out).all())

    def test_all_channels_dropped_is_stable(self):
        program = ns.SExpression(
            symbol="attention_amplitude",
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
        model = self.dsl.compute(self.dsl.initialize(program))
        out = model(torch.randn(2, self.input_dim))
        self.assertEqual(tuple(out.shape), (2, 6))
        self.assertTrue(torch.isfinite(out).all())
