import unittest

import neurosym as ns

dslf = ns.DSLFactory()
dslf.concrete("1", "() -> i -> i", lambda: 1)
dslf.concrete("+", "(#t -> i, #t -> i) -> #t -> i", lambda x, y: lambda t: x(t) + y(t))
dslf.concrete("id", "#a -> #a", lambda x: x)
dslf.concrete(
    "compose", "(#a -> #b, #b -> #c) -> #a -> #c", lambda f, g: lambda x: f(g(x))
)
dsl = dslf.finalize()


class TestEnumeratability(unittest.TestCase):
    def test_dsl_productions(self):
        expected = """
                1 :: () -> i -> i
                + :: (#t -> i, #t -> i) -> #t -> i
                id :: #a -> #a
        compose_0 :: (#a -> () -> () -> i, (() -> () -> i) -> #c) -> #a -> #c
        compose_1 :: (#a -> () -> i, (() -> i) -> #c) -> #a -> #c
        compose_2 :: (#a -> (i, i) -> i, ((i, i) -> i) -> #c) -> #a -> #c
        compose_3 :: (#a -> i -> i, (i -> i) -> #c) -> #a -> #c
        compose_4 :: (#a -> i, i -> #c) -> #a -> #c
        """
        actual = dsl.render()
        self.assertEqual(
            {line.strip() for line in actual.strip().split("\n")},
            {line.strip() for line in expected.strip().split("\n")},
        )

    def test_basic_enumerate(self):
        expans = {
            ns.render_s_expression(prog, False)
            for prog in dsl.expansions_for_type(
                ns.TypeWithEnvironment(ns.parse_type("i -> i"), ns.Environment.empty())
            )
        }
        print(expans)
        self.assertSetEqual(
            expans,
            {
                "(1)",
                "(id ??::<i -> i>)",
                "(+ ??::<i -> i> ??::<i -> i>)",
                "(compose_0 ??::<i -> () -> () -> i> ??::<(() -> () -> i) -> i>)",
                "(compose_1 ??::<i -> () -> i> ??::<(() -> i) -> i>)",
                "(compose_2 ??::<i -> (i, i) -> i> ??::<((i, i) -> i) -> i>)",
                "(compose_3 ??::<i -> i -> i> ??::<(i -> i) -> i>)",
                "(compose_4 ??::<i -> i> ??::<i -> i>)",
            },
        )
