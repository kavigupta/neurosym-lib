import unittest

import neurosym as ns


def basic_drop_dsl():

    import neurosym as ns

    dslf = ns.DSLFactory()

    dslf.concrete("+", "(f, f) -> f", lambda x, y: x + y)
    dslf.lambdas(include_drops=True, max_env_depth=5)
    dsl = dslf.finalize(prune_to="(f, f, f, f, f) -> f")

    return dsl


class TestEvaluateDrops(unittest.TestCase):

    def test_basic_drop(self):
        pass
