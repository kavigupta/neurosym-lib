"""
Check that all neural modules have desired input-output signatures.
"""
import unittest

import torch

from neurosym.examples.near.models.mlp import MLP, MLPConfig
from neurosym.examples.near.models.rnn import RNNConfig, Seq2ClassRNN, Seq2SeqRNN


class TestNeuralModels(unittest.TestCase):
    input_dim = 10
    output_dim = 4
    hidden_dim = 20
    traj_len = 3
    bs = 2

    def test_mlp(self):
        cfg = MLPConfig("mlp", self.input_dim, self.hidden_dim, self.output_dim)
        mlp = MLP(cfg)
        x = torch.randn(self.bs, self.traj_len, self.input_dim)
        self.assertEqual(mlp(x).shape, (self.bs, self.traj_len, self.output_dim))

    def test_seq2class_rnn(self):
        cfg = RNNConfig("rnn", self.input_dim, self.hidden_dim, self.output_dim)
        seq2class_rnn = Seq2ClassRNN(cfg)
        x = torch.randn(self.bs, self.traj_len, self.input_dim)
        self.assertEqual(seq2class_rnn(x).shape, (self.bs, self.output_dim))

    def test_seq2seq_rnn(self):
        cfg = RNNConfig("rnn", self.input_dim, self.hidden_dim, self.output_dim)
        seq2seq_rnn = Seq2SeqRNN(cfg)
        x = torch.randn(self.bs, self.traj_len, self.input_dim)
        self.assertEqual(
            seq2seq_rnn(x).shape, (self.bs, self.traj_len, self.output_dim)
        )
