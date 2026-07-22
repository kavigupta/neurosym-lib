"""Fast repro of the abstraction type-inference path in multicoreEnumeration.

The full experiment spends minutes on OCaml enumeration + recognition training
before it ever reaches the per-task abstraction-parsing loop in
``dreamcoder/enumeration.py``. But that loop is pure Python and depends ONLY on
the abstraction strings in the learned grammar -- not on any likelihoods,
frontiers, or timing. So we replay exactly that conversion here against the
abstractions DreamCoder already discovered (captured from the checkpoint), which
runs in well under a second.

Iterate on the fix in neurosym.compression.process_abstraction /
DSL.compute_type_abs, then rerun:

    python3 repro_enum_abstraction.py

Abstraction source (iteration 0 stitch output):
  experiments_iterative/outputs/list/domains/list/dreamcoder/seed_111/
      dreamcoder_95/0/train/stitch_compress_output.json  -> abstractions[*].dreamcoder
"""

import json
import os

from neurosym.compression.process_abstraction import (
    abstraction_production_from_dreamcoder,
)
from neurosym.examples.dreamcoder.list_example import list_dslf
from dreamcoder.domains.list.listPrimitives import LIST_OUTPUT_TYPES

STITCH_OUTPUT = (
    "experiments_iterative/outputs/list/domains/list/dreamcoder/"
    "seed_111/dreamcoder_95/0/train/stitch_compress_output.json"
)


def load_abstractions():
    """DreamCoder-format abstraction strings the run learned (checkpoint), with a
    hand-written fallback covering the known shapes if the checkpoint is absent.
    """
    if os.path.exists(STITCH_OUTPUT):
        with open(STITCH_OUTPUT) as f:
            data = json.load(f)
        return [a["dreamcoder"] for a in data["abstractions"]]
    return [
        "#(lambda (lambda (lambda ($1 ($1 $2 $2) $2))))",  # higher-order -> skip
        "#(lambda (all_0 (lambda $0) (++ empty empty)))",  # inner lambda
        "#(lambda (* (* 0 0) 1))",
        "#(lambda (++ (index $0 empty) empty))",
        "#(lambda (eq? (sum $0) 0))",
        "#(lambda (all_1 (lambda (is-prime $0)) $0))",  # inner lambda
    ]


def main():
    all_output_types = sorted(LIST_OUTPUT_TYPES)
    abstractions = load_abstractions()
    print(f"Replaying {len(abstractions)} learned abstraction(s)\n")

    typed, skipped = 0, 0
    for k in abstractions:
        # Fresh DSL per abstraction, matching the real per-task loop.
        dsl = list_dslf(*all_output_types).finalize()
        prod = abstraction_production_from_dreamcoder(dsl, str(k), str(k))
        if prod is None:
            print(f"[skip]  {k}\n        -> not representable/typable in NeuroSym DSL")
            skipped += 1
        else:
            print(f"[typed] {k}\n        -> {prod.type_signature().render()}")
            typed += 1

    print("\n" + "=" * 72)
    print(f"{typed} typed, {skipped} skipped, {len(abstractions)} total.")
    print("No crashes: every abstraction was either typed or cleanly skipped.")


if __name__ == "__main__":
    main()
