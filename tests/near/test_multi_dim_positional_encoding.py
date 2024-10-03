import unittest

import numpy as np
import torch

import neurosym as ns
from neurosym.examples import near


class TestMultiDimPositionalEncoding(unittest.TestCase):
    def test_no_sequential_axes(self):
        pe = near.BasicMultiDimensionalPositionalEncoding(6)
        # batch axis = 3, sequence axes absent, tensor size = 6
        x1, x2, x3 = torch.randn(7, 6), torch.randn(7, 6), torch.randn(7, 6)
        xenc = pe(ns.TypeShape((7,), ()), [x1, x2, x3])
        self.assertEqual(xenc.shape, (7, 3, 6))
        self.assertTrue(torch.allclose(xenc[:, 0], x1 + pe.pe[0]))
        self.assertTrue(torch.allclose(xenc[:, 1], x2 + pe.pe[1]))
        self.assertTrue(torch.allclose(xenc[:, 2], x3 + pe.pe[2]))

    def test_single_sequential_axis(self):
        pe = near.BasicMultiDimensionalPositionalEncoding(6)
        # batch axis = 3, sequence axis = 2, tensor size = 6
        x1 = torch.randn(7, 2, 6)

        xenc = pe(ns.TypeShape((7,), (2,)), [x1])

        self.assertEqual(xenc.shape, (7, 2, 6))

        self.assertTrue(
            torch.allclose(xenc[:, 0], x1[:, 0] + pe.pe[0] + pe.pe[0] @ pe.orthonormal)
        )
        self.assertTrue(
            torch.allclose(xenc[:, 1], x1[:, 1] + pe.pe[0] + pe.pe[1] @ pe.orthonormal)
        )

    def test_multiple_sequential_axis(self):
        pe = near.BasicMultiDimensionalPositionalEncoding(6)
        # batch axis = 3, sequence axes = [2, 3, 5], then 9, tensor size = 6
        x1 = torch.randn(7, 2, 3, 5, 6)
        x2 = torch.randn(7, 9, 6)
        xenc = pe(ns.TypeShape((7,), (2, 3, 5)), [x1, x2])
        self.assertEqual(xenc.shape, (7, 2 * 3 * 5 + 9, 6))
        idxs = np.arange(2 * 3 * 5).reshape(2, 3, 5)
        for i in range(2):
            for j in range(3):
                for k in range(5):
                    self.assertTrue(
                        torch.allclose(
                            xenc[:, idxs[i, j, k]],
                            x1[:, i, j, k]
                            + pe.pe[0]
                            + pe.pe[i] @ pe.orthonormal
                            + pe.pe[j] @ pe.orthonormal @ pe.orthonormal
                            + pe.pe[k]
                            @ pe.orthonormal
                            @ pe.orthonormal
                            @ pe.orthonormal,
                            rtol=0.001,
                        ),
                        f"Failed at {i}, {j}, {k}",
                    )
        for i in range(9):
            self.assertTrue(
                torch.allclose(
                    xenc[:, 2 * 3 * 5 + i],
                    x2[:, i] + pe.pe[1] + pe.pe[i] @ pe.orthonormal,
                    rtol=0.001,
                ),
                f"Failed at {i}",
            )
