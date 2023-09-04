import unittest

from neurosym.examples.basic_arith import basic_arith_dsl, int_type
from neurosym.types.type import AtomicType


class TestAllRules(unittest.TestCase):
    def test_basic_arith(self):
        self.assertEqual(
            {
                AtomicType(name="int"): [
                    ("+", [AtomicType(name="int"), AtomicType(name="int")]),
                    ("1", []),
                ]
            },
            basic_arith_dsl.all_rules(int_type),
        )
