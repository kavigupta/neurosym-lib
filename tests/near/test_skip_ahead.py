import unittest

import neurosym as ns


class TestSkipAhead(unittest.TestCase):

    def dslGraph(self, dsl, target_type):
        return ns.DSLSearchGraph(
            dsl,
            ns.parse_type(target_type),
            ns.ChooseFirst(),
            lambda x: True,
            ns.NoMetadataComputer(),
            skip_ahead=True,
        )

    def basicSkipAheadDSL(self):
        dslf = ns.DSLFactory()
        dslf.production("a1Tob", "a1 -> b", None)
        dslf.production("a2Tob", "a2 -> b", None)
        dslf.production("bToc", "b -> c", None)
        dslf.production("cTod", "c -> d", None)
        dslf.production("dToe", "d -> e", None)
        dslf.production("eTof", "e -> f", None)
        dslf.production("fTog", "f -> g", None)
        dslf.production("f2Tog", "f2 -> g", None)
        return dslf.finalize()

    def assertNodes(self, nodes, expected_reprs):
        result = sorted(ns.render_s_expression(n.program) for n in nodes)
        print(result)
        self.assertEqual(
            result,
            sorted(expected_reprs),
        )

    def test_do_skip(self):

        g = self.dslGraph(self.basicSkipAheadDSL(), "g")

        self.assertNodes(
            [g.initial_node()],
            ["??::<g>"],
        )

        self.assertNodes(
            g.expand_node(g.initial_node()),
            ["(fTog (eTof (dToe (cTod (bToc ??::<b>)))))", "(f2Tog ??::<f2>)"],
        )

    def deadEndLoopDSL(self):
        # ??<[a] -> [a]> expands to either
        #   (map ??::<a -> a>) which is fine
        #   (loop ??::<[a] -> b>, ??::<[a] -> [a]>). The latter is unconstructible
        #       but not obviously so because [a] -> b is constructible via loop.
        # Really the only thing that needs to exist here is loop, but the rest are
        #    necessary to prevent pruning.
        # We want to ensure no crash, it's fine if unconstructible nodes are presented,
        #   but they should not be expanded to infinity.
        dslf = ns.DSLFactory()

        dslf.production("id", "() -> a -> a", None)
        dslf.production("loop", "(#a -> b, #a -> #b) -> #a -> #b", None)
        dslf.production("map", "(#a -> #b) -> [#a] -> [#b]", None)
        dslf.production("aToB", "() -> a -> b", None)

        dslf.prune_to("[a] -> [a]")

        return dslf.finalize()

    def test_dead_end_loop(self):
        g = self.dslGraph(self.deadEndLoopDSL(), "[a] -> [a]")

        self.assertNodes(
            [g.initial_node()],
            ["??::<[a] -> [a]>"],
        )

        self.assertNodes(
            g.expand_node(g.initial_node()),
            [
                "(loop (loop (loop (loop (loop (loop (loop (loop (loop (loop (loop ??::<[a] -> b> ??::<[a] -> b>) ??::<[a] -> b>) ??::<[a] -> b>) ??::<[a] -> b>) ??::<[a] -> b>) ??::<[a] -> b>) ??::<[a] -> b>) ??::<[a] -> b>) ??::<[a] -> b>) ??::<[a] -> b>) ??::<[a] -> [a]>)",
                "(map ??::<a -> a>)",
            ],
        )
