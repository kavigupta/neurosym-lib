from dataclasses import dataclass

from neurosym.examples.tower.dsl import parse_sugared
from neurosym.programs.s_expression import SExpression


@dataclass
class DreamcoderTowerTask:
    name: str
    program: SExpression

    @classmethod
    def of(cls, name, program):
        return cls(name, parse_sugared(program))


def dreamcoder_tower_tasks():
    arches = [
        DreamcoderTowerTask.of(
            "arch leg %d" % n,
            f"(/seq (for ({n}) (lam (v))) (r (4)) (for ({n}) (lam (v))) (l (2)) (h))",
        )
        for n in range(1, 9)
    ]
    archesStacks = [
        DreamcoderTowerTask.of(
            "arch stack %d" % n,
            f"(for ({n}) (lam (/seq (v) (r (4)) (v) (l (2)) (h) (l (2)))))",
        )
        for n in range(3, 7)
    ]
    Bridges = [
        DreamcoderTowerTask.of(
            "bridge (%d) of arch %d" % (n, l),
            f"""
            (for ({n}) (lam (/seq
                (for ({l}) (lam (/seq (v) (r (4)) (v) (l (4)))))
                (r (2))
                (h)
                (r (4))
            )))
            """,
        )
        for n in range(2, 8)
        for l in range(1, 6)
    ]
    offsetArches = [
        DreamcoderTowerTask.of(
            "bridge (%d) of arch, spaced %d" % (n, l),
            f"""
            (for ({n}) (lam (/seq
                (embed (/seq (v) (r (4)) (v) (l (2)) (h)))
                (r ({l}))
            )))
            """,
        )
        for n, l in [(3, 7), (4, 8)]
    ]

    staircase1 = [
        DreamcoderTowerTask.of(
            "R staircase %d" % n,
            f"""
            (for ({n}) (lam (/seq
                (for ($0_0) (lam (/seq
                    (embed (/seq
                        (v) (r (4)) (v) (l (2)) (h)
                    ))
                )))
                (r (6))
            )))
            """,
        )
        for n in range(3, 8)
    ]
    staircase2 = [
        DreamcoderTowerTask.of(
            "L staircase %d" % n,
            f"""
            (for ({n}) (lam (/seq
                (for ($0_0) (lam (/seq
                    (embed (/seq
                        (v) (r (4)) (v) (l (2)) (h)
                    ))
                )))
                (l (6))
            )))
            """,
        )
        for n in range(3, 8)
    ]
    simpleLoops = [
        DreamcoderTowerTask.of(
            "%s row %d, spacing %d" % (o, n, s),
            f"""
            (for ({n}) (lam (/seq
                ({o}) (r ({s}))
            )))
            """,
        )
        for o, n, s in [("h", 4, 7), ("v", 5, 3)]
    ]

    pyramids = []
    pyramids += [
        DreamcoderTowerTask.of(
            "arch pyramid %d" % n,
            f"""
            (/seq
                (for ({n}) (lam (/seq
                    (for ($0_0) (lam (/seq
                        (embed (/seq
                            (v) (r (4)) (v) (l (2)) (h)
                        ))
                    )))
                    (r (6))
                )))
                (for ({n}) (lam (/seq
                    (for (- ({n}) ($0_0)) (lam (/seq
                        (embed (/seq
                            (v) (r (4)) (v) (l (2)) (h)
                        ))
                    )))
                    (r (6))
                )))
            )
            """,
        )
        for n in range(2, 6)
    ]
    pyramids += [
        DreamcoderTowerTask.of(
            "H pyramid %d" % n,
            f"""
            (/seq
                (for ({n}) (lam (/seq
                    (for ($0_0) (lam (/seq
                        (h)
                    )))
                    (r (6))
                )))
                (for ({n}) (lam (/seq
                    (for (- ({n}) ($0_0)) (lam (/seq
                        (h)
                    )))
                    (r (6))
                )))
            )
            """,
        )
        for n in range(4, 6)
    ]
    pyramids += [
        DreamcoderTowerTask.of(
            "H 1/2 pyramid %d" % n,
            f"""
            (for ({n}) (lam (/seq
                (r (6))
                (embed (/seq
                    (for ($0_0) (lam (/seq
                        (h) (l (3))
                    )))
                ))
            )))
            """,
        )
        for n in range(4, 8)
    ]
    pyramids += [
        DreamcoderTowerTask.of(
            "arch 1/2 pyramid %d" % n,
            f"""
            (for ({n}) (lam (/seq
                (r (6))
                (embed (/seq
                    (for ($0_0) (lam (/seq
                        (embed (/seq
                            (v) (r (4)) (v) (l (2)) (h)
                        ))
                        (l (3))
                    )))
                ))
            )))
            """,
        )
        for n in range(2, 8)
    ]
    bricks = [
        DreamcoderTowerTask.of(
            "brickwall, %dx%d" % (w, h),
            f"""
            (for ({h}) (lam (/seq
                (embed (/seq
                    (for ({w}) (lam (/seq
                        (h) (r (6))
                    )))
                ))
                (embed (/seq
                    (r (3))
                    (for ({w}) (lam (/seq
                        (h) (r (6))
                    )))
                ))
            )))
            """,
        )
        for w in range(3, 7)
        for h in range(1, 6)
    ]
    aqueducts = [
        DreamcoderTowerTask.of(
            "aqueduct: %dx%d" % (w, h),
            f"""
            (for ({w}) (lam (/seq
                {"(v) " * h}
                (r (4))
                {"(v) " * h}
                (l (2)) (h) (l (2)) (v) (r (4)) (v) (l (2)) (h) (r (4))
            )))
            """,
        )
        for w in range(4, 8)
        for h in range(3, 6)
    ]

    compositions = (
        [
            DreamcoderTowerTask.of(
                "%dx%d-bridge on top of %dx%d bricks" % (b1, b2, w1, w2),
                f"""
                (/seq
                    (for ({w1}) (lam (/seq
                        (embed (/seq
                            (for ({w2}) (lam (/seq
                                (h) (r (6))
                            )))
                        ))
                        (embed (/seq
                            (r (3))
                            (for ({w2}) (lam (/seq
                                (h) (r (6))
                            )))
                        ))
                    )))
                    (r (1))
                    (for ({b1}) (lam (/seq
                        (for ({b2}) (lam (/seq
                            (v) (r (4)) (v) (l (4))
                        )))
                        (r (2)) (h) (r (4))
                    )))
                )
                """,
            )
            for b1, b2, w1, w2 in [(5, 2, 4, 5)]
        ]
        + [
            DreamcoderTowerTask.of(
                "%d pyramid on top of %dx%d bricks" % (p, w1, w2),
                f"""
                (/seq
                    (for ({w1}) (lam (/seq
                        (embed (/seq
                            (for ({w2}) (lam (/seq
                                (h) (r (6))
                            )))
                        ))
                        (embed (/seq
                            (r (3))
                            (for ({w2}) (lam (/seq
                                (h) (r (6))
                            )))
                        ))
                    )))
                    (r (1))
                    (for ({p}) (lam (/seq
                        (for ($0_0) (lam (/seq
                            (embed (/seq
                                (v) (r (4)) (v) (l (2)) (h)
                            ))
                        )))
                        (r (6))
                    )))
                    (for ({p}) (lam (/seq
                        (for (- ({p}) ($0_0)) (lam (/seq
                            (embed (/seq
                                (v) (r (4)) (v) (l (2)) (h)
                            ))
                        )))
                        (r (6))
                    )))
                )
                """,
            )
            for w1, w2, p in [(2, 5, 2)]
        ]
        + [
            DreamcoderTowerTask.of(
                "%d tower on top of %dx%d bricks" % (t, w1, w2),
                f"""
                (/seq
                    (for ({w1}) (lam (/seq
                        (embed (/seq
                            (for ({w2}) (lam (/seq
                                (h) (r (6))
                            )))
                        ))
                        (embed (/seq
                            (r (3))
                            (for ({w2}) (lam (/seq
                                (h) (r (6))
                            )))
                        ))
                    )))
                    (r (6))
                    {"(v) " * t}
                    (r (4))
                    {"(v) " * t}
                    (l (2))
                    (h)
                )
                """,
            )
            for t, w1, w2 in [(4, 1, 3)]
        ]
    )

    everything = (
        arches
        + simpleLoops
        + Bridges
        + archesStacks
        + aqueducts
        + offsetArches
        + pyramids
        + bricks
        + staircase2
        + staircase1
        + compositions
    )
    return everything
