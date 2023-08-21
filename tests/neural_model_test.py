"""
Check that shapes match for defined neural models.
"""

import unittest
from neurosym.models.mlp import MLP, MLPConfig
from neurosym.models.rnn import Seq2ClassRNN, Seq2SeqRNN, RNNConfig
import torch


class TestNeuralModels(unittest.TestCase):
    def test_mlp(self):
        cfg = MLPConfig("mlp", 10, 20, 4)
        mlp = MLP(cfg)
        x = torch.randn(2, 3, 10)
        self.assertEqual(mlp(x).shape, (2, 3, 4))

    def test_seq2class_rnn(self):
        cfg = RNNConfig("rnn", 10, 20, 4)
        seq2class_rnn = Seq2ClassRNN(cfg)
        x = torch.randn(2, 3, 10)
        self.assertEqual(seq2class_rnn(x).shape, (2, 4))

    def test_seq2seq_rnn(self):
        cfg = RNNConfig("rnn", 10, 20, 4)
        seq2seq_rnn = Seq2SeqRNN(cfg)
        x = torch.randn(2, 3, 10)
        self.assertEqual(seq2seq_rnn(x).shape, (2, 3, 4))
