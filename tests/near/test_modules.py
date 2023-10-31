"""
Check that all neural modules have desired input-output signatures.
"""
import unittest

import torch

import neurosym as ns


class TestNeuralModels(unittest.TestCase):
    input_dim = 10
    output_dim = 4
    hidden_dim = 20
    traj_len = 3
    bs = 2

    def test_mlp(self):
        cfg = ns.examples.near.MLPConfig(
            "mlp", self.input_dim, self.hidden_dim, self.output_dim
        )
        mlp = ns.examples.near.MLP(cfg)
        x = torch.randn(self.bs, self.traj_len, self.input_dim)
        self.assertEqual(mlp(x).shape, (self.bs, self.traj_len, self.output_dim))

    def test_seq2class_rnn(self):
        cfg = ns.examples.near.RNNConfig(
            "rnn", self.input_dim, self.hidden_dim, self.output_dim
        )
        seq2class_rnn = ns.examples.near.Seq2ClassRNN(cfg)
        x = torch.randn(self.bs, self.traj_len, self.input_dim)
        self.assertEqual(seq2class_rnn(x).shape, (self.bs, self.output_dim))

    def test_seq2seq_rnn(self):
        cfg = ns.examples.near.RNNConfig(
            "rnn", self.input_dim, self.hidden_dim, self.output_dim
        )
        seq2seq_rnn = ns.examples.near.Seq2SeqRNN(cfg)
        x = torch.randn(self.bs, self.traj_len, self.input_dim)
        self.assertEqual(
            seq2seq_rnn(x).shape, (self.bs, self.traj_len, self.output_dim)
        )
