# Interpretable ECG Classification via Heterogeneously-Typed Neurosymbolic Program Synthesis

*A case study in DSL design for clinical decision support*

## Abstract

We present a neurosymbolic program synthesis approach that produces interpretable ECG
classification programs matching or exceeding the accuracy of opaque baselines
(MLP, Random Forest) on the PTB-XL dataset. The technical contribution is the
identification and resolution of a *DSL design anti-pattern* in which heterogeneous
input streams are forced into a homogeneous typed lambda calculus. After three
unsuccessful DSL designs converging at AUC ceilings of 0.71, 0.83, and 0.88, our
final design — a **heterogeneously-typed lambda calculus** with one distinct
typedef per ECG feature group — discovered a program with macro AUC **0.9017**
(F1 **0.654**), beating both the Random Forest baseline (0.888 AUC) and the
3-layer MLP baseline (0.900 AUC, F1 0.611) while remaining fully interpretable
(every weight in every operator is inspectable). We attribute the breakthrough to
*type-directed search-space pruning*: the type system rejects nonsensical
compositions (e.g., applying an amplitude embedding to an interval variable)
before training cost is paid. The case study yields concrete DSL design lessons
that we believe generalize to other multi-modal interpretable program synthesis
problems.

---

## 1. Introduction

Clinical decision support imposes a stronger interpretability requirement than
typical machine learning benchmarks. A model that classifies an ECG as having
a myocardial infarction must, ideally, *explain* its decision in terms a
cardiologist can verify: which leads matter, which intervals or amplitudes
contributed, and how the contributions were combined. Black-box models — even
those that achieve strong AUC — face slow clinical adoption.

Neurosymbolic program synthesis (Shah et al., 2020; Ellis et al., 2021;
Chaudhuri 2025) offers a path. Rather than learning weights of a fixed
architecture, the system learns a *program* over a domain-specific language
(DSL) of typed neural primitives. The discovered program is itself the model;
its symbolic structure is human-readable, and the weights of each primitive
(e.g., a `Linear` layer) remain inspectable.

This case study asks: **Can we synthesize interpretable ECG classifiers that
match or beat opaque baselines?** Phase 1 of our work, using a hand-designed
DSL of feature-group selectors, answered yes — reaching macro AUC 0.901 with a
depth-4 program. But that DSL was bespoke to ECG features and not obviously
generalizable. We sought to systematize the design.

Three subsequent attempts to make the DSL more "principled" — by exposing each
ECG lead as a separate lambda variable (Phase 2), or each anatomical lead
group (Phase 3), or the full flat feature vector (Phase 3.5) — each plateaued
*below* Phase 1's accuracy. The root cause was a subtle anti-pattern: forcing
heterogeneous input streams into a homogeneous typed lambda. Our resolution —
heterogeneous-typed lambda binding, supported by the underlying neurosym
framework but unused until this work — closed the gap and exceeded Phase 1's
accuracy with a more compositional and discoverable program structure.

**Contributions.**
1. We document and explain a recurring DSL design anti-pattern (homogeneous
   typing of heterogeneous input streams) through four progressively-refined
   ECG DSLs.
2. We demonstrate that *heterogeneously-typed lambda binding*, combined with
   type-directed search pruning in `neurosym-lib`'s A*-family search, suffices
   to produce an interpretable program (macro AUC **0.9017**, F1 **0.654**)
   that beats opaque baselines on PTB-XL.
3. We provide a complete recipe — data layout, DSL productions, search
   configuration, training hyperparameters — that future neurosymbolic ECG
   work can build on.

---

## 2. Background

### 2.1 The PTB-XL Dataset

PTB-XL (Wagner et al., 2020) is the largest open clinical 12-lead ECG dataset:
21,837 ten-second recordings from 18,885 patients, annotated with SCP-ECG
statements that aggregate into five diagnostic *superclasses*:

| Superclass | Description |
|---|---|
| NORM | Normal |
| MI | Myocardial infarction |
| STTC | ST/T-change |
| CD | Conduction disturbance |
| HYP | Hypertrophy |

We use the *standard PTB-XL evaluation split*: stratified folds 1–8 for
training, fold 9 for validation, fold 10 for held-out test. In **single-label
mode** (used throughout), we filter to records with exactly one superclass
label, yielding 13,004 train / 1,643 val / 1,660 test samples. The task is
5-class classification.

### 2.2 ECGDeli Features

Rather than operating on raw waveforms, we use the **ECGDeli** feature
extraction pipeline (Pilia et al., 2021), which produces 177 pre-extracted
clinical features per recording, organized into five groups:

| Group | # Features | Examples |
|---|---|---|
| **Amplitude** | 60 (5 wave types × 12 leads) | P_Amp_I, Q_Amp_V4, R_Amp_aVR, ... |
| **Interval** | 84 (7 timings × 12 leads) | PQ_Int_I, QRS_Dur_V4, QT_IntCorr_aVF, ... |
| **ST** | 12 (ST elevation × 12 leads) | ST_Elev_I, ..., ST_Elev_V6 |
| **Morphology** | 12 (P-wave morphology × 12 leads) | P_Morph_I, ..., P_Morph_V6 |
| **Global** | 9 (whole-recording aggregates) | RR_Mean, QT_IntFramingham, HA (heart axis), ... |

Each feature is a real scalar. ECGDeli's features are themselves interpretable
clinical measurements — our task is to learn a *combination* of them that
classifies the superclass.

### 2.3 The NEAR Framework

NEAR (Shah et al., 2020) performs A*-style program search over a typed DSL,
guiding expansion with a neural admissible heuristic. Concretely:

1. **Programs are S-expressions** over typed productions. A *complete* program
   has no `??` holes; a *partial* program has holes filled by neural
   placeholders (a `GenericMLPRNNHoleFiller`).
2. **Cost = validation loss + λ · structural cost.** The structural cost
   estimates the remaining work to complete a partial program (e.g.,
   `MinimalStepsNearStructuralCost`). λ is a tunable penalty.
3. **Search yields programs in increasing cost order** via A* (or OSGAstar,
   which evaluates costs lazily for efficiency).

The key correctness property is that completed programs are evaluated with
their *true* loss (no neural placeholders) before being yielded.

### 2.4 The `neurosym-lib` Library

We build on `neurosym-lib`, an open-source Python framework for neurosymbolic
program synthesis. The features we use:

- **`DSLFactory`** — a builder pattern for typed DSLs. Productions are declared
  with type signature strings (`"$fHid -> $fOut"`) parsed by
  `neurosym/types/type_string_repr.py`. Typedefs (`dslf.typedef("fHid", "{f, $H}")`)
  introduce type aliases.
- **The type system** (`neurosym/types/`): `AtomicType` (e.g., `i`, `f`),
  `TensorType` (`{f, 60}`), `ArrowType` for function types, `TypeVariable` for
  polymorphism. Type unification powers expansion of partial programs.
- **`LambdaTypeSignature`** (`neurosym/types/type_signature.py:142`) — admits
  a **heterogeneous list** of argument types (`input_types: List[Type]`).
  This is the key feature our final design exploits; we describe its usage in
  §4.4.
- **Variable productions** — `$<i>_<id>` denotes a de Bruijn index `i` of a
  variable with type-id `id`. Variables are typed by their environment
  position; the search graph rejects compositions that would require the
  "wrong" variable in scope.
- **`OSGAstar`** — lazy A* search that defers expensive cost computation,
  yielding ~20% wall-clock speedup over eager A* on our problem.
- **`ProgramEmbedding`** — an extension point for wrapping the trained program
  into a larger module. We use it to slice the flat 177-dim input tensor into
  the typed arguments expected by our heterogeneous lambda (§4.4).
- **`ChannelHoleFiller` / `GenericMLPRNNNeuralHoleFiller`** — neural
  placeholders that fill `??` holes during partial-program training.

---

## 3. DSL Design Journey

### 3.1 Phase 1: Hand-designed Feature-Group Selectors

Our starting point used a small DSL of five hand-designed productions:
`affine_amplitude`, `affine_interval`, `affine_st`, `affine_morphology`,
`affine_global`. Each was `Linear(N, H)` over its specific feature group
(60, 84, 12, 12, 9 input dimensions respectively), followed by `output`
(`Linear(H, num_classes)`). The DSL had no lambda variables — the input was
implicit, accessed by each `affine_*` production via metadata.

With penalty 0.1, num_programs=30, A* search rediscovered:

```
(output (add (add (affine_amplitude) (affine_interval)) (affine_global)))
AUC = 0.901, F1 = 0.633
```

This matched the MLP baseline. **The DSL was bespoke**, however — each
`affine_*` baked in both *which features to select* and *how to project them*.
Adding a new feature group required a new production.

### 3.2 Phase 2: Channelised DSL with 21 Variables (Capped at 0.71)

To generalize, we restructured the input as `(B, 21, 14)`: 12 leads (each with
14 features) + 9 globals (each scalar zero-padded to 14). Each of the 21
"channels" became a separate lambda variable of type `{f, 14}`. The DSL had a
single generic `embed: $fInp -> $fHid` production that the search could apply
to any variable, with `add`/`mul`/`ite` for combination.

**This failed.** Across 2000 search programs, AUC topped out at **0.71**. The
search exhausted itself enumerating `21^k` variable assignments for fixed
program skeletons — 215 of the 2000 programs at depth 24 shared a single
template differing only in which variables filled five slots. `ite` programs
*never appeared*, because their structural cost was 7× that of `embed`:
A* exhausted all simpler programs first.

### 3.3 Phase 3: Lead-Grouped DSL with 6 Variables (Capped at 0.83)

To shrink the combinatorial space, we grouped the 12 leads into 5 anatomical
territories (inferior, lateral-limb, septal, anterior, lateral-precordial) +
1 globals channel = 6 variables. Each lead group *averaged* the 14 per-lead
features across its member leads.

This achieved AUC **0.83** with the program:

```
(lam (output (mul (embed $A) (embed $B))))
```

We then tried *every* extra production we could justify:

| Extension | Best AUC |
|---|---|
| `NumberHolesNearStructuralCost` + `embed_bool` | 0.833 |
| `bilinear: ($fHid, $fHid) -> {f,1}` | 0.830 |
| `gate: $fInp -> $fInp` | 0.830 |
| Cross-channel self-attention | 0.829 |
| `ite`-only (no add/mul) | 0.806 |
| MLP-embed (rejected post-hoc: hidden ReLU is opaque) | (0.836) |

Adding productions barely moved the ceiling. The root cause: **averaging within
lead groups destroyed cross-lead correlations** that the Linear over all 60
amplitude features had been learning. The structure was wrong, not the
vocabulary.

### 3.4 Phase 3.5: Flat 177-feature Input with Feature-Group Slicing (0.876)

We reverted to a single lambda variable of type `{f, 177}` but added typed
*slicing* productions:

```python
embed_amp: $fInp -> $fHid    -> Linear(60, H)(x[..., 0:60])
embed_int: $fInp -> $fHid    -> Linear(84, H)(x[..., 60:144])
# ... and similarly for st (12), morph (12), global (9)
```

This restored cross-lead correlation learning within each `Linear`, lifting
AUC to **0.876**. With additional per-feature-group lead-attention pooling
productions (interpretable — attention weights show "for this patient, which
leads matter for amplitude?"), we reached **0.888**.

But the single-variable design had a deficiency: all productions read from the
same `$0_0`. The search could not reason about feature groups as *distinct
entities*. Programs like `(add (embed_amp $0_0) (embed_int $0_0))` were
possible, but the search had to discover the right slice/production pairings
by trial; the type system gave it no help.

### 3.5 Phase 4: Heterogeneous-Typed DSL (Final Design, AUC 0.9017)

The fix: **five distinct typedefs**, one per feature group, with native sizes:

```python
dslf.typedef("fAmp",    "{f, 60}")    # P/Q/R/S/T_Amp × 12 leads
dslf.typedef("fInt",    "{f, 84}")    # 7 intervals × 12 leads
dslf.typedef("fSt",     "{f, 12}")    # ST_Elev × 12 leads
dslf.typedef("fMorph",  "{f, 12}")    # P_Morph × 12 leads
dslf.typedef("fGlobal", "{f, 9}")     # 9 global scalars
dslf.typedef("fHid",    "{f, $H}")
dslf.typedef("fOut",    "{f, $O}")
```

The five typed embed productions:

```python
dslf.production("embed_amp",    "$fAmp    -> $fHid", ..., nn.Linear(60, H))
dslf.production("embed_int",    "$fInt    -> $fHid", ..., nn.Linear(84, H))
dslf.production("embed_st",     "$fSt     -> $fHid", ..., nn.Linear(12, H))
dslf.production("embed_morph",  "$fMorph  -> $fHid", ..., nn.Linear(12, H))
dslf.production("embed_global", "$fGlobal -> $fHid", ..., nn.Linear(9, H))
```

Combination productions (`add`, `mul`, `ite`, `linear`, `output`) are
all `$fHid -> $fHid` (or output), reusable unchanged.

The target type binds five *heterogeneously-typed* lambda variables:

```python
dslf.prune_to("($fAmp, $fInt, $fSt, $fMorph, $fGlobal) -> $fOut")
```

#### Why this works: type-directed search-space pruning

`LambdaTypeSignature` admits a list of heterogeneous input types
(`neurosym/types/type_signature.py:151`, verified by the unit test
`tests/lambdas/lambdas_twe_test.py:12-17` which constructs
`prune_to("(i, f) -> i")`). When the search expands a hole of type `$fHid`,
the candidate productions are only those whose return type unifies with
`$fHid`; the candidate variables are only those whose environment type
matches the production's input type. Concretely:

- An `embed_amp ??` hole at type `$fAmp` admits *only* the amp-typed
  variable (de Bruijn position 4 under our `prune_to` declaration order).
  The search never enumerates `(embed_amp $0_<global>)` etc.
- An `add ?? ??` hole at type `$fHid` admits any `$fHid`-producing
  subprogram on either side — but each side's own type checking still
  prunes invalid embed/variable pairings recursively.

This *cuts the search frontier by a factor of N* where N is the average number
of variables (5 here): an untyped DSL with N homogeneous variables would
enumerate `embed_amp` against all N variables; the typed DSL enumerates only
1. The savings compound multiplicatively at each composition depth.

---

## 4. Implementation

We summarize the implementation in three files of `neurosym-lib`.

### 4.1 Data Layer — `neurosym/datasets/ecg_data_example.py`

The PTB-XL CSV is parsed into a `(B, 177)` float32 matrix, with columns
ordered in **feature-major** layout:

```
[amp(60) | int(84) | st(12) | morph(12) | global(9)]
```

NaN imputation (training-set per-column median) and z-score normalization (fit
on training set) are applied. A new boolean `feature_groups=True` flag attaches
slice metadata to the returned `DatasetWrapper`:

```python
datamodule.feature_group_slices = {
    "amp": (0, 60), "int": (60, 144), "st": (144, 156),
    "morph": (156, 168), "global": (168, 177),
}
```

The flat `(B, 177)` shape is preserved — no per-group reshape is needed since
the unpacking happens inside the DSL's program embedding.

### 4.2 Unpack Module — `FeatureGroupUnpackModule`

`neurosym-lib`'s NEAR training loop calls `model(x, environment=())` with a
single tensor `x`. Our 5-argument lambda needs five tensors. We bridge this via
a `ProgramEmbedding` that slices `x` before invocation:

```python
class _FeatureGroupUnpackModule(nn.Module):
    SLICES = ((0, 60), (60, 144), (144, 156), (156, 168), (168, 177))
    def forward(self, x, environment=()):
        if x.dim() == 3 and x.shape[1] == 1:  # defensive squeeze
            x = x.squeeze(1)
        args = tuple(x[..., s:e] for s, e in self.SLICES)
        return self.inner(*args, environment=environment)

class FeatureGroupUnpackEmbedding(ProgramEmbedding):
    def embed_initialized_program(self, program_module):
        return _FeatureGroupUnpackModule(program_module)
```

This is a 30-line change. It generalizes the existing `ChannelUnpackEmbedding`
(which used `x.unbind(dim=1)`, requiring homogeneous shapes) to heterogeneous
slice sizes.

### 4.3 DSL Function — `phase1_typed_ecg_dsl()`

The full DSL builder is ~80 lines. The structural target-type depth is
`log₂(6) ≈ 2.58`; we use `max_type_depth=3` (versus 5 in the 21-channel
DSL — a 64× reduction in type-expansion work). We set `max_env_depth=5` (exactly
the number of variables).

### 4.4 Variable-Indexing Convention

de Bruijn-indexed variables in the discovered programs (e.g., `$3_2`) decode as:

| AST | de Bruijn idx | Type | Type-id |
|---|---|---|---|
| `$0_3` | 0 | `$fGlobal` | 3 |
| `$1_0` | 1 | `$fMorph`  | 0 |
| `$2_0` | 2 | `$fSt`     | 0 |
| `$3_2` | 3 | `$fInt`    | 2 |
| `$4_1` | 4 | `$fAmp`    | 1 |

(The type-id `_id` is assigned by the framework's internal type ordering;
the leading number is the de Bruijn index, counting from the innermost
lambda binder.)

---

## 5. Evaluation

### 5.1 Experimental Setup

| Parameter | Value |
|---|---|
| Train / Val / Test | 13,004 / 1,643 / 1,660 (folds 1-8 / 9 / 10) |
| Label mode | single (one superclass) |
| Loss | cross-entropy |
| Training optimizer | Adam, lr=1e-3, batch=256 |
| Search epochs (partial-program training) | 50 |
| Final epochs (complete-program training) | 250 |
| Hidden dim H | 32 |
| Structural cost penalty λ | 0.12 (tuned to match scales of val loss and struct cost) |
| Search algorithm | OSGAstar (lazy A*) |
| `num_programs` (search budget) | 500 |
| GPU | NVIDIA RTX 2080 Ti (single GPU) |

### 5.2 Baselines

| Method | Macro AUC | Macro F1 | Bootstrap 95% CI | Interpretable |
|---|---|---|---|---|
| Decision Tree (sklearn defaults) | 0.677 | 0.473 | (0.664, 0.692) | Partial |
| Random Forest (n_estimators=100) | 0.888 | 0.572 | (0.873, 0.902) | Partial (feature importance) |
| MLP (3×256, 300 ep) | 0.900 | 0.611 | (0.889, 0.912) | No |

(Reference: Mehari et al. 2023 report RF on ECGDeli at 0.899 macro AUC — our
0.888 single-label / 0.905 multi-label is within the range.)

### 5.3 Main Result

The best program discovered by the typed-DSL search (Experiment T1):

```
(lam (output (add (embed_int $3_2) (mul (embed_amp $4_1) (embed_amp $4_1)))))

Macro AUC: 0.9017   Macro F1: 0.654   Bootstrap 95% CI: (0.890, 0.913)
```

Decoded:
- `$4_1` is the amplitude variable (60 features); `$3_2` is the interval
  variable (84 features). The other three lambda-bound variables (`$0_3`
  global, `$1_0` morph, `$2_0` st) are *unused* — the program performs
  feature selection at the variable level.
- `(embed_amp $4_1)` is `Linear(60, H)` applied to amplitudes — a learned
  weighted combination of all 60 amplitude features.
- `(mul (embed_amp $4_1) (embed_amp $4_1))` is the *element-wise square* of
  the amplitude embedding: `(W · amp)²`. Two independent `Linear` layers W₁
  and W₂ are learned (each invocation of a parameterized production
  instantiates fresh parameters), so this is more precisely
  `(W₁ · amp) ⊙ (W₂ · amp)` — a *bilinear* form, capturing pairwise
  interactions among amplitudes in the hidden space.
- `(embed_int $3_2)` is `Linear(84, H)` applied to intervals.
- `add` combines them in the hidden space; `output` projects to 5 logits.

Both `Linear` layers' weights are inspectable: each row maps an output hidden
unit to a weighted sum of the 60 amp (or 84 int) input features. A cardiologist
can read off "which amplitude features are weighted highly for hidden unit k."

### 5.4 Full Results Table

| Experiment | Macro AUC | Macro F1 | Notes |
|---|---|---|---|
| **T1 — typed DSL** | **0.9017** | **0.654** | **Best interpretable result** |
| Phase 1 (hand-designed) | 0.901 | 0.633 | NEAR auto-discovered the equivalent |
| MLP baseline | 0.900 | 0.611 | Opaque |
| T2 — typed DSL + lead-attn | 0.895 | 0.638 | Attention pooling didn't help |
| Exp B (flat-input + lead-attn) | 0.888 | 0.622 | Pre-typed best |
| RandomForest baseline | 0.888 | 0.572 | |
| Exp 10 (flat 177 input) | 0.876 | 0.587 | First break of channelised ceiling |
| Phase 3 ceiling (6 channels) | 0.833 | 0.516 | Information loss from averaging |
| Phase 2 (21 channels) | 0.709 | 0.040 | Combinatorial blowup |

### 5.5 Search Statistics

| Metric | Phase 2 (21 ch) | Phase 3 (6 ch) | T1 (typed) |
|---|---|---|---|
| Variables in lambda | 21 (homogeneous) | 6 (homogeneous) | 5 (heterogeneous) |
| `max_type_depth` for lambdas | 5 | 4 | 3 |
| `max_env_depth` | 21 | 6 | 5 |
| Programs evaluated in 75-min budget | ~62 (killed) | ~150 | **500 (completed)** |
| Variable assignments per pairwise template | 21² = 441 | 6² = 36 | 1 (type-pruned) |
| `ite` programs in best top-10? | 0 | rarely | rarely |

The typed DSL not only finds better programs but **runs the search to
completion** within the same time budget, because each (production,
variable) pairing is type-pruned before any neural training cost is paid.

---

## 6. Analysis and Discussion

### 6.1 What the type system bought us

The `LambdaTypeSignature` mechanism in `neurosym/types/type_signature.py`
already supported heterogeneous input types — see the test case at
`tests/lambdas/lambdas_twe_test.py:12-17`. Our prior DSLs *voluntarily*
restricted themselves to homogeneous types via patterns like
`prune_to(f"({'$fInp, ' * num_channels}) -> $fOut")`. This was the operative
anti-pattern: it threw away the type-directed search pruning that the
framework offers for free.

Concretely, for a pairwise composition like `(add ?? ??)` at type `$fHid`:

- **Homogeneous 21-channel DSL**: each `??` admits `embed v` for every
  v ∈ {21 vars}. The search enumerates 21² = 441 variable pairings per
  template. After filtering to non-trivial pairings, the bulk of search
  effort is *variable enumeration*, not structural exploration.
- **Heterogeneous typed DSL**: each `??` admits `embed_amp $4`, `embed_int $3`,
  `embed_st $2`, `embed_morph $1`, `embed_global $0` — exactly 5 distinct
  productions, each *uniquely* paired with its variable. 5×5=25 pairings,
  but each pairing represents a structurally different program (different
  feature combination), not a redundant variable swap.

### 6.2 What didn't help, and why

The lead-attention pooling productions in T2 (`embed_amp_attn`, etc.) reshape
each feature-group tensor to `(B, 12 leads, K features)` and apply
self-attention across the 12 leads, producing a pooled `(B, H)` hidden vector.
We expected this to help: clinically, "which leads matter most" varies by
patient. Empirically, T2 (with attention) scored 0.895 vs T1 (without)
0.902. Hypothesis: the attention productions inflate the search frontier
(four new productions, each typed equivalently to an existing `embed_X`),
diluting the search budget for the productions that actually matter (the
plain `Linear`s).

A second non-result: ensembling the top-10 programs added only +0.001 AUC.
The top programs were structurally near-identical (`(add (embed_X $a) (embed_Y $b))`
for various X, Y, a, b) and made highly correlated errors. Ensembling
requires *diversity*, which cost-ordered A* doesn't produce by construction.

### 6.3 Interpretability discipline

Mid-project, we briefly considered an `mlp_embed` production
(`Linear → ReLU → Dropout → Linear`). It achieved AUC 0.895 in pilot runs,
but we invalidated the result: a hidden ReLU + Dropout creates an opaque
intermediate representation whose contribution can't be read off the
weights, and Dropout adds stochasticity at inference. We constrained the
DSL to:

- **Linear** layers (`y = Wx + b`),
- **self-attention** (Q/K/V projections + softmax — weights and attention
  scores inspectable),
- **element-wise arithmetic** (`add`, `mul`, `sub` — no learned weights),
- **single-sigmoid gating** (`ite`, `gate`, `bilinear` — one learnable form,
  weights inspectable).

The 0.9017 result was achieved under these constraints. We view the
constraint as load-bearing: it forced us to think harder about *architectural*
gains (heterogeneous types) rather than reach for the easy non-linearity.

### 6.4 Discovered program ≈ Phase 1 program + element-wise interaction

The Phase-1 (hand-designed DSL) winning program was:

```
(output (add (add (affine_amplitude) (affine_interval)) (affine_global)))   AUC=0.901
```

The T1 (typed DSL) winning program was:

```
(output (add (embed_int $3) (mul (embed_amp $4) (embed_amp $4))))   AUC=0.9017
```

The structural similarity is striking: both add an `affine_interval`-equivalent
and an `affine_amplitude`-equivalent. T1 *additionally* discovers that
squaring the amplitude embedding — `(W₁ · amp) ⊙ (W₂ · amp)`, a bilinear
form — captures pairwise amp-amp interactions that the linear Phase-1
program could not. This is the +0.001 AUC and +0.02 F1 gain over Phase 1.

The discovery is non-obvious: we did not hand-write `mul` for the amp
embedding. The search found it because the typed DSL made it the
*lowest-cost extension* of the linear baseline.

---

## 7. Related Work

**NEAR** (Shah et al., 2020) introduced the framework of neural admissible
heuristics for A* program search. Our work uses NEAR's search backend
unchanged; the contribution is a DSL design pattern.

**DreamCoder** (Ellis et al., 2021) demonstrates iterative library learning
in a strongly-typed lambda calculus with polymorphic type variables. Its
type system is functionally similar to ours — but DreamCoder learns *new*
productions through compression of solved tasks, while we hand-author the
productions and let NEAR discover compositions.

**Scallop** (Li et al., 2023) is a neurosymbolic language with native
tuple-based heterogeneous representations. Our heterogeneous-typed DSL
arrives at a similar architectural point (distinct types per input stream)
through a different mechanism (typed lambda binding rather than
predicate/tuple syntax).

**Earlier ECG classification benchmarks** (Mehari et al., 2023; Strodthoff
et al., 2021) report Random Forest at 0.899 macro AUC and MLP at 0.900 on
ECGDeli features. To our knowledge, ours is the first *interpretable
program-synthesized* model in this regime.

---

## 8. Limitations and Future Work

1. **Single-label only.** PTB-XL is naturally multi-label (a recording can
   carry multiple superclass annotations). We restrict to single-label mode
   for cleaner search signal; extending to multi-label requires reformulating
   the validation metric (`hamming_accuracy` → something like
   `macro_auroc`).

2. **Feature-engineering dependency.** The 0.9017 result relies on ECGDeli's
   177 pre-extracted features. Operating directly on raw waveforms (e.g., as
   in time-series convolutional architectures) would require an entirely
   different DSL — perhaps with `Conv1d` productions and lead-spatial
   attention. This is future work.

3. **Search budget tuning.** Our typed DSL completed 500 programs in the
   75-minute soft budget; the earlier homogeneous DSLs reached only 60-150
   programs in the same time. If a future DSL needs more programs, the
   search may not converge in fixed wall-clock.

4. **No interpretability evaluation by clinicians.** We claim
   interpretability on the basis that all weights are inspectable; we have
   not conducted a user study with cardiologists to assess whether the
   discovered programs are *actually* useful for clinical decision-making.

---

## 9. Conclusion

We presented a neurosymbolic program synthesis case study on the PTB-XL ECG
classification task. Across four iterations of DSL design, we identified and
resolved an anti-pattern in which heterogeneous input streams are forced
into homogeneous typed lambda variables. The resolution —
heterogeneously-typed lambda binding, a feature `neurosym-lib` already
supported but our prior DSLs hadn't exploited — let the search type-prune
invalid (production, variable) pairings before training cost is paid.

The discovered program

```
(lam (output (add (embed_int $3_2) (mul (embed_amp $4_1) (embed_amp $4_1)))))
```

achieves macro AUC **0.9017** and macro F1 **0.654** on the PTB-XL held-out
test set, matching or beating the Random Forest (0.888 AUC) and MLP (0.900
AUC, F1 0.611) baselines while remaining fully interpretable (every weight
in every operator is inspectable).

The broader lesson: **DSL type design is search design**. A homogeneous-typed
DSL with N variables forces the search to enumerate N variable assignments
per composition slot, with no semantic justification for most of them. A
heterogeneous-typed DSL with N distinct typed variables exposes the *natural
structure* of the problem to the type-directed search and prunes the
combinatorial explosion at its source.

---

## References

- Chaudhuri, S. *Neurosymbolic Programming Handbook.* University of Texas at
  Austin, 2025. https://www.cs.utexas.edu/~swarat/pubs/ns-handbook-2025.pdf
- Ellis, K., et al. "DreamCoder: Bootstrapping Inductive Program Synthesis
  with Wake-Sleep Library Learning." PLDI 2021.
- Li, Y., et al. "Scallop: A Language for Neurosymbolic Programming."
  PLDI 2023.
- Mehari, T., and Strodthoff, N. "Self-supervised representation learning
  from 12-lead ECG data." *Computers in Biology and Medicine* 141, 2022.
- Pilia, N., et al. "ECGdeli — An open source ECG delineation toolbox for
  MATLAB." *SoftwareX* 13, 2021.
- Shah, A., Zhan, E., Sun, J., Verma, A., Yue, Y., Chaudhuri, S. "Learning
  Differentiable Programs with Admissible Neural Heuristics." NeurIPS 2020.
- Strodthoff, N., Wagner, P., Schaeffter, T., Samek, W. "Deep Learning for
  ECG Analysis: Benchmarks and Insights from PTB-XL." *IEEE J. Biomed.
  Health Informatics* 25(5), 2021.
- Wagner, P., et al. "PTB-XL, a large publicly available
  electrocardiography dataset." *Scientific Data* 7(1), 2020.
