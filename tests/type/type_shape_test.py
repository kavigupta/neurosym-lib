import unittest

import neurosym as ns


class TestComputeTypeShapeTensor(unittest.TestCase):

    def test_exact_tensor_direct_works(self):
        self.assertEqual(
            ns.compute_type_shape(ns.parse_type("{f, 24, 32}"), (24, 32)),
            ns.TypeShape(batch_size=(), sequence_lengths=()),
        )

    def test_tensor_type_different(self):
        self.assertEqual(
            ns.compute_type_shape(ns.parse_type("{abc, 24, 32}"), (24, 32)),
            ns.TypeShape(batch_size=(), sequence_lengths=()),
        )

    def test_tensor_type_bad(self):
        with self.assertRaises(ns.TypeShapeException) as e:
            ns.compute_type_shape(ns.parse_type("{[f], 24, 32}"), (24, 32))

        self.assertEqual(
            str(e.exception),
            "Expected TensorType to have an atomic element typ, but received [f]",
        )

    def test_exact_tensor_direct_not_enough_dims(self):
        with self.assertRaises(ns.TypeShapeException) as e:
            ns.compute_type_shape(ns.parse_type("{f, 24, 32}"), (24,))
        self.assertEqual(
            str(e.exception),
            "Too few dimensions for type {f, 24, 32}: expected 2, got 1: (24,)",
        )

    def test_exact_tensor_direct_mismatch(self):
        with self.assertRaises(ns.TypeShapeException) as e:
            ns.compute_type_shape(ns.parse_type("{f, 24, 32}"), (24, 32, 1))

        self.assertEqual(
            str(e.exception),
            "In type {f, 24, 32}, expected the 0th dimension to be 24, but received 32 instead: (24, 32, 1)",
        )

    def test_exact_tensor_direct_extra_dims(self):
        self.assertEqual(
            ns.compute_type_shape(ns.parse_type("{f, 24, 32}"), (20, 34, 24, 32)),
            ns.TypeShape(batch_size=(20, 34), sequence_lengths=()),
        )


class TestComputeTypeShapeSequence(unittest.TestCase):
    def test_exact_sequence_direct_works(self):
        self.assertEqual(
            ns.compute_type_shape(ns.parse_type("[f]"), (5, 2)),
            ns.TypeShape(batch_size=(5,), sequence_lengths=(2,)),
        )

    def test_multiple_sequences(self):
        self.assertEqual(
            ns.compute_type_shape(ns.parse_type("[[f]]"), (9, 5, 2, 10)),
            ns.TypeShape(batch_size=(9, 5), sequence_lengths=(2, 10)),
        )

    def test_tensor_in_sequence(self):
        self.assertEqual(
            ns.compute_type_shape(ns.parse_type("[{f, 10}]"), (5, 2, 10)),
            ns.TypeShape(batch_size=(5,), sequence_lengths=(2,)),
        )

    def test_arrow_in_sequence(self):
        with self.assertRaises(ns.TypeShapeException) as e:
            ns.compute_type_shape(ns.parse_type("[f -> f]"), (5, 2))
        self.assertEqual(
            str(e.exception),
            "Cannot compute shape of f -> f",
        )


class TestApplyTypeShape(unittest.TestCase):
    def test_no_addtl_dims(self):
        self.assertEqual(
            ns.TypeShape(batch_size=(), sequence_lengths=()).apply(
                ns.parse_type("{f, 24, 32}")
            ),
            (24, 32),
        )

    def test_with_batch_dims(self):
        self.assertEqual(
            ns.TypeShape(batch_size=(2, 5), sequence_lengths=()).apply(
                ns.parse_type("{f, 24, 32}")
            ),
            (2, 5, 24, 32),
        )

    def test_with_sequence_dims(self):
        self.assertEqual(
            ns.TypeShape(batch_size=(), sequence_lengths=(2, 5)).apply(
                ns.parse_type("[[{f, 3}]]")
            ),
            (2, 5, 3),
        )


class TestInferOutputShape(unittest.TestCase):
    def test_basic_usage(self):
        self.assertEqual(
            ns.infer_output_shape(
                [ns.parse_type("[{f, 10}]"), ns.parse_type("[{f, 2}]")],
                [(30, 5, 10), (30, 5, 2)],
                ns.parse_type("[{f, 36}]"),
            ),
            (ns.TypeShape(batch_size=(30,), sequence_lengths=(5,)), (30, 5, 36)),
        )

    def test_transfers_to_different_num_tensor_dims(self):
        self.assertEqual(
            ns.infer_output_shape(
                [ns.parse_type("[{f, 10}]"), ns.parse_type("[{f, 2}]")],
                [(30, 5, 10), (30, 5, 2)],
                ns.parse_type("[{f, 36, 2}]"),
            ),
            (ns.TypeShape(batch_size=(30,), sequence_lengths=(5,)), (30, 5, 36, 2)),
        )

    def test_inconsistent_input_shapes(self):
        with self.assertRaises(ns.TypeShapeException) as e:
            ns.infer_output_shape(
                [ns.parse_type("[{f, 10}]"), ns.parse_type("[{f, 2}]")],
                [(30, 5, 10), (30, 8, 2)],
                ns.parse_type("[{f, 36}]"),
            )

        self.assertEqual(
            str(e.exception),
            "Inconsistent sequence lengths: [(5,), (8,)]",
        )

    def test_inconsistent_batch_sizes(self):
        with self.assertRaises(ns.TypeShapeException) as e:
            ns.infer_output_shape(
                [ns.parse_type("[{f, 10}]"), ns.parse_type("[{f, 2}]")],
                [(30, 5, 10), (40, 5, 2)],
                ns.parse_type("[{f, 36}]"),
            )

        self.assertEqual(
            str(e.exception),
            "Inconsistent batch sizes: [(30,), (40,)]",
        )

    def test_too_many_sequence_dimensions(self):
        with self.assertRaises(ns.TypeShapeException) as e:
            ns.infer_output_shape(
                [ns.parse_type("[[[[[{f, 10}]]]]]")],
                [(1, 2, 3, 4, 5, 10)],
                ns.parse_type("[[{f, 36, 2}]]"),
            )

        self.assertEqual(
            str(e.exception),
            "Too many sequence dimensions, [[{f, 36, 2}]] expected 2, got 5",
        )

    def test_not_enough_sequence_dimensions(self):
        with self.assertRaises(ns.TypeShapeException) as e:
            ns.infer_output_shape(
                [ns.parse_type("[[{f, 10}]]")],
                [(1, 2, 10)],
                ns.parse_type("[[[[[{f, 36, 2}]]]]]"),
            )

        self.assertEqual(
            str(e.exception),
            "Too few sequence dimensions, [[[[[{f, 36, 2}]]]]] expected 5, got 2",
        )
