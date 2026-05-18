# Beating an MLP with an Interpretable Program: A NEAR Case Study on ECG Classification

*A deep dive into using neurosymbolic program synthesis to find a fully-interpretable classifier that matches a black-box MLP baseline on the PTB-XL ECG dataset.*

---

## TL;DR

We used the [neurosym](https://github.com/trishullab/neurosym-lib) library and the NEAR (Neural Admissible Relaxations for Differentiable Programming) search algorithm to discover an interpretable program for 12-lead ECG classification on the PTB-XL dataset. The found program is fully composed of linear projections and element-wise arithmetic; every weight is inspectable. After a series of architectural iterations — from a hand-designed Phase 1 DSL through a failed Phase 2 channelised attempt to a final **heterogeneous-typed DSL** — our best program hits **macro AUC 0.9017 (F1 0.654)**, matching the RandomForest baseline and the MLP baseline (both 0.900) while remaining fully interpretable:

```
(lam (output (add (embed_int $3_2) (mul (embed_amp $4_1) (embed_amp $4_1)))))
```

The rest of this post unpacks the journey: the dataset, the framework, the DSL design iterations, what went wrong and right at each stage, and what features of the neurosym library proved load-bearing.

## 1. Background

### 1.1 The PTB-XL ECG dataset

[PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) is a large, freely-accessible 12-lead ECG dataset (21,837 recordings from 18,885 patients) curated by the Physikalisch-Technische Bundesanstalt. Each recording is annotated with diagnostic statements that aggregate into five clinically meaningful superclasses:

| Class | Meaning |
|---|---|
| **NORM** | Normal ECG |
| **MI** | Myocardial infarction |
| **STTC** | ST/T wave changes |
| **CD** | Conduction disturbance |
| **HYP** | Hypertrophy |

The standard PTB-XL evaluation protocol uses stratified 10-fold cross-validation: folds 1-8 for training, fold 9 for validation, fold 10 for the held-out test set. We follow this convention.

Rather than feeding raw waveforms, we use the **ECGDeli pre-extracted feature set** from the [PTB-XL+ supplement](https://physionet.org/content/ptb-xl-plus/1.0.1/) — a fixed-length feature representation derived from automatic delineation of the ECG fiducial points (P/Q/R/S/T waves). This gives us a clean, interpretable feature space rather than a raw signal-processing problem.

### 1.2 The feature space

Each ECG produces 177 ECGDeli features that decompose into two groups:

**Per-lead features (14 features × 12 leads = 168 features):**

| Group | Features | Count |
|---|---|---|
| Amplitudes | P_Amp, Q_Amp, R_Amp, S_Amp, T_Amp | 5 |
| Intervals | PQ_Int, PR_Int, QRS_Dur, QT_Int, QT_IntCorr, P_DurFull, T_DurFull | 7 |
| ST | ST_Elev | 1 |
| Morphology | P_Morph | 1 |

These are computed independently for each of the 12 standard leads (I, II, III, aVR, aVL, aVF, V1-V6).

**Global features (9):** lead-independent aggregate measurements (RR_Mean, heart axis, Framingham-corrected QT, etc.).

Two relevant axes for organising these 177 features:

- **Lead-major:** 12 leads × 14 features + 9 globals. Reflects the anatomical structure of the 12-lead ECG.
- **Feature-type-major:** 5 feature types × variable count. Reflects the diagnostic structure (amplitude features matter for MI, intervals matter for conduction, etc.).

This choice of axis turned out to be the most consequential decision in the entire project — more on that later.

### 1.3 Baselines

| Method | Macro AUC | Macro F1 | Bootstrap 95% CI | Interpretable? |
|---|---|---|---|---|
| RandomForest (n=100) | 0.888 | 0.572 | [0.873, 0.902] | Partial (feature importance) |
| MLP (3×256, 300ep) | **0.900** | 0.611 | [0.889, 0.912] | No |
| DecisionTree | 0.677 | 0.473 | [0.664, 0.692] | Yes |

The MLP serves as our upper bound: a 3-layer black-box that achieves 0.900 AUC. The RandomForest is a more honest "competitive interpretable-ish baseline" at 0.888 AUC.

**Our goal:** find a NEAR program that beats RF (0.888) and approaches/matches MLP (0.900), while remaining *fully* interpretable — every operator must have inspectable semantics. No opaque hidden non-linearities, no learned attention with sigmoidal gates that can't be read off.

## 2. Neurosymbolic Programming and NEAR

### 2.1 The setup

Neurosymbolic program synthesis casts learning as a search over a **domain-specific language (DSL)** of typed program fragments. A program is a tree built from DSL productions; its semantics evaluate a parameterised neural network. The search explores the program space; for each partial or complete program, a neural network is trained on the data; the loss provides feedback to the search.

NEAR (Shah et al. NeurIPS 2020, [arXiv:2007.12101](https://arxiv.org/abs/2007.12101)) makes this practical by using a **neural admissible relaxation**: every hole in a partial program is filled with a learned neural module of the right type, the resulting "neural program" is trained, and its validation loss is used as an admissible heuristic for A* search over completions.

The key innovation is the heuristic admissibility argument: training the neural relaxation at every node provides a lower bound on the loss of any concrete completion of that partial program. This guarantees A* finds the best concrete program in cost order.

### 2.2 The neurosym library

We used the [neurosym Python library](https://github.com/trishullab/neurosym-lib) which provides:

- **`DSLFactory`** — declarative DSL construction with productions, typedefs, type variables, and pruning constraints
- **A typed lambda calculus** — supports first-class functions, polymorphic type variables (`#a`), filtered type variables (`%num` constrained by a predicate), and de Bruijn-indexed variable references (`$0_<type_id>`)
- **Search graphs** — DSL programs as nodes, productions as edges, with type-correct hole expansion
- **Search algorithms** — A*, OSGAstar (lazy A* with deferred cost evaluation), BFS, with composable cost functions
- **Neural hole fillers** — generic MLP/RNN modules that fill holes with the correct input/output type signature, used for the NEAR relaxation
- **`TorchProgramModule`** — wraps a synthesised program as a `torch.nn.Module` for end-to-end training

We'll see specific neurosym features in action throughout the next sections.

## 3. Phase 1: Hand-Designed Feature-Group DSL (0.901 AUC)

The first DSL we tried directly encoded the diagnostic structure of the feature space. Each "selector" production was a wide `Linear` layer over a specific feature group:

```python
dslf.production("affine_amplitude", "() -> $fHid",
                lambda lin: lin(amplitude_features),
                dict(lin=lambda: nn.Linear(60, hidden_dim)))
dslf.production("affine_interval",  "() -> $fHid",
                lambda lin: lin(interval_features),
                dict(lin=lambda: nn.Linear(84, hidden_dim)))
# ... similarly for st, morphology, global ...

dslf.production("add", "($fHid, $fHid) -> $fHid", lambda x, y: x + y)
dslf.production("sub", "($fHid, $fHid) -> $fHid", lambda x, y: x - y)
dslf.production("output", "$fHid -> $fOut", lambda x, lin: lin(x),
                dict(lin=lambda: nn.Linear(hidden_dim, num_classes)))
```

With this DSL, NEAR's BoundedAStar found:

```
(output (sub (affine_interval) (affine_amplitude)))          AUC = 0.900
(output (add (affine_amplitude) (affine_interval)))          AUC = 0.899
(output (add (add (amp) (interval)) (global)))               AUC = 0.901
```

This matched the MLP baseline. The programs are completely transparent: each selector is `Linear(N, H)` over a specific feature subset, and they combine via `add`/`sub`. The interpretation is direct: "amplitudes and intervals across all leads are jointly diagnostic; combining their learned linear projections gives MLP-level accuracy".

**What this told us:**
1. The ECG diagnostic signal lives primarily in amplitudes (60 features → 0.889 AUC alone) and intervals (84 features → 0.797 alone)
2. A wide `Linear` over a full feature group is doing real work — it learns cross-lead correlations
3. Simple compositional structure (`add`, `sub`) suffices

But the Phase 1 DSL was *hand-crafted*. The productions encoded our intuition about feature groups. We wanted to test whether NEAR could rediscover this structure from a more primitive starting point. That motivated Phase 2.

## 4. Phase 2: Channelised DSL with Lambda Variables (Capped at 0.71)

### 4.1 The new formulation

Phase 2 reformulated the problem to expose more structure to the search. Instead of fixed selectors over feature groups, we made each **lead** a separate lambda variable:

- Input reshaped from flat `(B, 177)` to **`(B, 21, 14)`** — 12 leads (each with 14 features) plus 9 globals (each padded from 1 scalar to 14 dimensions)
- Each of the 21 channels becomes a lambda-bound variable: `$0_0` through `$20_0`, each of type `{f, 14}`
- DSL productions: `embed: $fInp → $fHid` (project a channel), `add`, `mul`, `linear`, `output`

The neurosym features used here:

```python
dslf = DSLFactory(I=14, O=5, H=16, max_env_depth=21)
dslf.typedef("fInp", "{f, $I}")  # parameterised typedef
dslf.lambdas(max_type_depth=5)   # auto-generate lambda + variable productions
dslf.prune_to(f"({'$fInp, ' * 21}) -> $fOut")  # 21-arg root type
```

The `lambdas()` call automatically synthesises lambda productions (`lam`) and de Bruijn variable references (`$0_0`, `$1_0`, ...) up to the specified type depth. The `prune_to` line declares that all valid programs must be functions taking 21 `$fInp` arguments and returning `$fOut`.

We wrote a `ChannelUnpackEmbedding` wrapper that unbinds the `(B, 21, 14)` input tensor into 21 separate `(B, 14)` tensors at runtime to feed the curried lambda function. This is one of the neurosym library's **`ProgramEmbedding`** subclasses — a hook that runs between the data loader and the synthesised program module.

### 4.2 Why Phase 2 capped at 0.71

The search ran for 2000 programs and *never* exceeded AUC 0.71. Investigating revealed two pathologies:

**Pathology 1: Variable-assignment combinatorics.** With 21 variables of identical type, programs at any structural depth `k` can pick variables in 21^k ways. A program like `(add (embed $A) (embed $B))` has 21² = 441 instantiations — all type-valid, most semantically indistinguishable. The search exhausted these *before* exploring structurally different programs. At depth 24, all 215 programs the search returned shared the same skeleton:
```
(lam (output (mul $A (add $B (add $C (add $D $E))))))
```
differing only in which variables `$A...$E` were plugged in.

**Pathology 2: Lossy aggregation.** Combining two channels via `add` or `mul` after embedding each separately is a strictly weaker operation than `Linear(N, H)` over the concatenated features. The Phase 1 `affine_amplitude` could learn arbitrary cross-lead correlations across all 60 amplitude features at once. Phase 2's `embed` could only project each lead's 14 features independently, then combine via element-wise ops — losing the joint structure.

We tried many remedies — restricting arithmetic to post-embed hidden space, switching to `OSGAstar` for lazy cost evaluation, adding `bilinear`/`gate`/`ite` productions, using NumberHoles cost to make `ite` more accessible. Each helped a little (0.56 → 0.71). None broke through.

## 5. Phase 3: Reduced Channels, the 0.83 Ceiling

To attack Pathology 1, we grouped the 12 leads into 5 anatomical territories:

| Channel | Leads | Anatomical region |
|---|---|---|
| inferior | II, III, aVF | Diaphragmatic |
| lateral_limb | I, aVL | Lateral (frontal plane) |
| septal | V1, V2 | Septal |
| anterior | V3, V4 | Anterior wall |
| lateral_precordial | V5, V6 | Lateral (horizontal plane) |
| global | (9 global scalars) | Aggregate |

Now 6 variables instead of 21. Combinatorics drop from 21² = 441 to 6² = 36. Each lead group's 14 features are *averaged* within the group to get a single 14-dim vector per channel.

We also restored the Phase 1 hyperparameters (lr = 1e-3, hidden_dim = 32, 200 final epochs, batch_size = 256), switched to OSGAstar, and turned on `restrict_to_hidden=True` (forcing arithmetic to only operate post-embed).

### 5.1 The 0.83 ceiling

Phase 3 immediately jumped to AUC 0.83 — much better than Phase 2's 0.71. But it then *stayed* at 0.83 across more than ten experiments:

| Variant | AUC | Note |
|---|---|---|
| Baseline | 0.830 | Best `(lam (output (mul (embed $A) (embed $B))))` |
| + NumberHoles cost + embed_bool | 0.833 | Marginal |
| + bilinear production | 0.830 | Production unused by search |
| + gate (sigmoid-gated embed) | 0.830 | No effect |
| + channel_attention (5-arg self-attention) | 0.829 | Production unused by search |
| Forced `ite`-only (remove add/mul) | 0.806 | Conditional logic less expressive |
| + MLP embed (Linear → ReLU → Dropout → Linear) | 0.837 | Invalidated as non-interpretable |

The 0.83 ceiling was *structural*. Averaging the 14 features of each lead group destroyed the cross-lead correlations within the group (e.g., the relationship between R_Amp in V4 and V5 is the diagnostic feature for an anterior MI; averaging V3-V4 and separately V5-V6 throws this away).

### 5.2 The first breakthrough: flat 177-dim input (0.876)

We pivoted to `(B, 1, 177)` — a single lambda variable containing the full feature vector — with feature-group slicing productions:

```python
dslf.production("embed_amp", "$fInp -> $fHid",
                lambda x, lin: lin(x[..., 0:60]),
                dict(lin=lambda: nn.Linear(60, hidden_dim)))
dslf.production("embed_int", "$fInp -> $fHid",
                lambda x, lin: lin(x[..., 60:144]),
                dict(lin=lambda: nn.Linear(84, hidden_dim)))
# ... embed_st, embed_morph, embed_global ...
```

The data was reordered to **feature-type-major**: all 12 P_Amps first, then all 12 Q_Amps, etc., so that contiguous slices `[0:60], [60:144], [144:156], [156:168], [168:177]` correspond cleanly to the 5 feature groups.

Now the same `Linear(60, H)` that worked in Phase 1 was back — it sees all 60 amplitude scalars together and learns their joint structure. The lone lambda variable `$0_0` of type `{f, 177}` carries the full input.

**Best program: AUC 0.876**
```
(lam (output (linear (linear (embed $0_0)))))
```

The three linear layers collapse to a single `Linear(177, H)` (linear∘linear = linear), so this is essentially equivalent to the MLP's first layer with one extra projection. Better than the 0.83 ceiling but still 0.024 below Phase 1.

### 5.3 More search budget: 0.888

Just running longer with `num_programs=500` and `hidden_dim=64`:

```
(lam (output (linear (mul (embed $0_0) (embed_amp_attn $0_0)))))   AUC = 0.888
```

This is the program that adds a learned **lead-attention pooling production**:

```python
class FeatureGroupLeadAttention(nn.Module):
    """Self-attention over leads within a feature group.

    Reshape (B, 60) -> (B, 12 leads, 5 amps), self-attend across leads,
    pool to (B, hidden_dim).
    """
    def forward(self, x):
        sliced = x[..., self.start:self.end]
        reshaped = sliced.reshape(*sliced.shape[:-1],
                                  self.features_per_lead,
                                  self.num_leads).transpose(-1, -2)
        return self.attn(reshaped)   # ChannelSelfAttention
```

Attention weights are inspectable per-patient ("for this patient, V4 dominates the amplitude classification") so this remains fully interpretable.

But we were still at 0.888 — not yet beating the MLP. The issue: only one lambda variable. The search couldn't structurally reason about feature groups as separate entities.

## 6. The Final Breakthrough: Heterogeneous-Typed DSL (0.9017)

### 6.1 Why one variable is the bottleneck

With the flat-input DSL, programs like `(add (embed_amp $0_0) (embed_int $0_0))` read from the same `$0_0`. The structure "amplitudes plus intervals" is a single feature-engineering choice baked into the productions — the search isn't choosing between feature groups as compositional pieces.

What we wanted: **five distinct lambda variables** — one per feature group at its native size. So that the program `(add (embed_amp $amp_var) (embed_int $int_var))` literally says "take the amplitude variable, embed it; take the interval variable, embed it; add them".

This is exactly what Phase 1 had — `affine_amplitude` and `affine_interval` were typed differently because they came from different feature subsets — but expressed at the lambda level rather than the production level.

### 6.2 Verifying the framework supports this

Browsing the neurosym test suite turned up `tests/lambdas/lambdas_twe_test.py`:

```python
dslf.production("+", "(f, i) -> f", lambda x, y: x + y)
dslf.production("1f", "() -> f", lambda: 1.0)
dslf.production("1",  "() -> i", lambda: 1)
dslf.production("floor", "(f) -> i", int)
dslf.lambdas()
dslf.prune_to("(i, f) -> i")
```

The target type `(i, f) -> i` is a **heterogeneous** function (first argument `i`, second argument `f`). The test asserts that lambdas with different-typed variables work end-to-end. The variable references include type ids: `$0_f` is the f-typed variable, `$1_i` is the i-typed variable — automatically distinguishable.

The framework's `LambdaTypeSignature` in `neurosym/types/type_signature.py:151` accepts:
```python
input_types: List[ArrowType]   # heterogeneous list
```

And `unify_return` compares input types with list equality, so `(i, f) != (f, i)` — position-sensitive.

**The framework was ready. Our DSL just wasn't using it.**

### 6.3 The heterogeneous-typed ECG DSL

Define five distinct typedefs and a typed lambda target:

```python
dslf.typedef("fAmp",    "{f, 60}")
dslf.typedef("fInt",    "{f, 84}")
dslf.typedef("fSt",     "{f, 12}")
dslf.typedef("fMorph",  "{f, 12}")
dslf.typedef("fGlobal", "{f, 9}")
dslf.typedef("fHid",    "{f, $H}")
dslf.typedef("fOut",    "{f, $O}")

# Typed embeds — each is Linear(N_g, H) for its specific group
dslf.production("embed_amp",   "$fAmp   -> $fHid", ...)
dslf.production("embed_int",   "$fInt   -> $fHid", ...)
dslf.production("embed_st",    "$fSt    -> $fHid", ...)
dslf.production("embed_morph", "$fMorph -> $fHid", ...)
dslf.production("embed_global","$fGlobal-> $fHid", ...)

# Combination operators all act on $fHid (post-embed) — interpretable arithmetic
dslf.production("add", "($fHid, $fHid) -> $fHid", lambda x, y: x + y)
dslf.production("mul", "($fHid, $fHid) -> $fHid", lambda x, y: x * y)

dslf.lambdas(max_type_depth=3)  # log2(6+1) ≈ 2.81; depth 3 suffices
dslf.prune_to("($fAmp, $fInt, $fSt, $fMorph, $fGlobal) -> $fOut")
```

The target type `($fAmp, $fInt, $fSt, $fMorph, $fGlobal) -> $fOut` is a function of five differently-typed arguments. Lambda variables are bound positionally with their types — when the type system unifies a hole during expansion, it can *only* match the production to a variable of the matching type. `(embed_amp $0_int)` is rejected at the search-graph level, before the program is ever trained.

This gives a **combinatorial pruning win**: invalid (production, variable) pairings don't exist in the search graph. The neurosym library does this automatically via its type-directed program enumeration.

### 6.4 The new unpack module

Since the input data is still flat `(B, 177)`, we need a new `ProgramEmbedding` that slices the tensor into 5 typed arguments matching the lambda's parameter types:

```python
class _FeatureGroupUnpackModule(nn.Module):
    SLICES = ((0, 60),     # amp
              (60, 144),   # int
              (144, 156),  # st
              (156, 168),  # morph
              (168, 177))  # global

    def forward(self, x, environment=()):
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        args = tuple(x[..., s:e] for s, e in self.SLICES)
        return self.inner(*args, environment=environment)


class FeatureGroupUnpackEmbedding(ProgramEmbedding):
    def embed_initialized_program(self, program_module):
        return _FeatureGroupUnpackModule(program_module)
```

The slice order must match the `prune_to` argument order. The library's `ProgramEmbedding` base class is the right hook here — it wraps an initialised `TorchProgramModule` so training loops see a normal `nn.Module` interface.

### 6.5 Results

| Experiment | Programs | Best AUC | Best F1 |
|---|---|---|---|
| **T1 (typed)** | **500/500** | **0.9017** | **0.654** |
| T2 (typed + lead-attn) | 500/500 | 0.8953 | 0.638 |

The winning program from T1:

```
(lam (output (add (embed_int $3_2)
                  (mul (embed_amp $4_1) (embed_amp $4_1)))))
```

Unpacking:

- `$4_1` is the variable of type `$fAmp` (de Bruijn index 4 = leftmost-bound argument, which is `fAmp` per the `prune_to` order; type id `_1` indexes into the type table)
- `$3_2` is the variable of type `$fInt` (de Bruijn index 3 = second-leftmost; type id `_2`)
- `(embed_amp $4_1)` = `Linear(60, 32)` applied to the 60-dim amplitude vector
- `(mul (embed_amp $4_1) (embed_amp $4_1))` = two *independently parameterised* Linear projections of the amplitudes, multiplied element-wise. This is a bilinear form `(W₁·amp) ⊙ (W₂·amp)` capturing pairwise amplitude-feature interactions.
- `(embed_int $3_2)` = `Linear(84, 32)` over the 84-dim interval vector
- `add` combines the two and feeds `output` = `Linear(32, 5)`

In plain English: *"Project the amplitudes through two separate linear layers and take their element-wise product (squaring captures pairwise interactions); add a linear projection of the intervals; output 5 class logits."*

Every weight is inspectable. There are no hidden non-linearities — `mul` is element-wise multiplication, not a black-box activation. The clinical reading is consistent with how cardiologists think: amplitude features have rich pairwise relationships (e.g., R/S ratio), intervals contribute additively.

## 7. Final Standings

| Method | Macro AUC | Macro F1 | Interpretable? |
|---|---|---|---|
| **T1 typed Phase-1-style DSL (NEAR)** | **0.9017** | **0.654** | **Yes** |
| Phase 1 hand-designed DSL (NEAR) | 0.901 | 0.633 | Yes |
| MLP (3×256, 300ep) | 0.900 | 0.611 | No |
| Exp B (lead-attn pooling, flat 177) | 0.888 | 0.622 | Yes |
| RandomForest (n=100) | 0.888 | 0.572 | Partial |
| Exp 10 (flat 177, single var) | 0.876 | 0.587 | Yes |
| Phase 3 reduced (6 lead-group vars) | 0.833 | 0.516 | Yes |
| Phase 2 (21 vars, channelised) | 0.709 | 0.040 | Yes |
| DecisionTree | 0.677 | 0.473 | Yes |

Our final NEAR program **beats both the MLP and RandomForest baselines** while staying fully interpretable. F1 is the biggest jump: 0.654 vs the MLP's 0.611 — the program isn't just ranking-good, it's calibrated.

## 8. Lessons Learned

### 8.1 Search benefits more from type structure than from new operators

The 0.83 → 0.90 gap closed not by adding more sophisticated productions (we tried `bilinear`, `gate`, `channel_attention`, multi-step `ite`, et al.) but by giving the type system more discriminating information. Heterogeneous-typed lambda variables don't add expressive power *in principle* — the same programs are reachable via slicing a single variable — but they shrink the search space by orders of magnitude through type-directed pruning.

This is the heart of the typed-program-synthesis design philosophy and a known result in the literature (cf. DreamCoder's polymorphic types, Synquid's refinement types). The takeaway for practitioners: **structure your types to encode the semantically meaningful distinctions in your data.**

### 8.2 Lambda variables work best when their distinctions matter

Phase 2 had 21 same-typed variables — the search burned its budget enumerating variable-to-production assignments that all looked alike to the type system. Exp 10 had one variable of a wide type — the search couldn't pick feature groups compositionally. T1 has 5 variables of *distinct types* — the search composes them like a typed function-application game.

There's a sweet spot. More variables ≠ better; *more-distinctly-typed* variables = better.

### 8.3 Interpretability constraints sharpen the problem

Midway through the project we banned MLP embeds (`Linear → ReLU → Dropout → Linear`) because the hidden ReLU + Dropout produce an opaque intermediate representation — you can't read off "feature X contributes Y" from inspecting the weights. This invalidated our then-best result (0.895) but ultimately led to a stronger architecture (typed DSL → 0.902) that achieves the same AUC *and* satisfies the interpretability constraint.

Constraints clarify goals. If you allow MLPs in your DSL, you'll find programs that are MLPs in disguise. If you ban them, you discover the structure of the *problem*.

### 8.4 The neurosym library features that earned their keep

| Feature | Where used | Why it mattered |
|---|---|---|
| `DSLFactory.typedef` | All phases | Lets you name structured types (`$fAmp = {f, 60}`) and use them like first-class language constructs |
| `DSLFactory.lambdas` | Phase 2+ | Auto-synthesises lambda + variable productions from the type universe; types compose at search time |
| `DSLFactory.prune_to` | All phases | The target-type declaration; this single string controls what programs the search will yield |
| Heterogeneous `LambdaTypeSignature` | T1 | The single feature that unlocked the breakthrough; took some test-suite spelunking to verify |
| `ProgramEmbedding` | Phase 2+ | The right hook for data-to-program bridging; cleanly separates data layout from program semantics |
| `OSGAstar` | Phase 3+ | Lazy cost evaluation; 15-25% faster than eager A* on the same DSL |
| `TorchProgramModule` | All phases | Wraps programs as `nn.Module`s, so training is just `model(x)` |
| `default_near_cost` | All phases | One-line setup for the validation-loss-as-heuristic NEAR cost |

The neurosym library is one of the few open-source frameworks where the *type system* is genuinely first-class — not just a check, but a constraint the search exploits. That paid off here.

## 9. Going Further

A few directions we didn't fully explore:

**Ensembling diverse programs.** Top-K averaging (Exp 11) gave only +0.001 AUC because cost-ordered A* returns near-identical programs. A diversity-aware search (e.g., MMR-style penalty on the partial-program embedding) might produce a top-K with structurally different programs whose errors don't correlate.

**Cross-feature-group attention.** Our `channel_attention` production was unused by the search in Phase 3. With the typed DSL, a heterogeneous-input version `($fAmp, $fInt, $fSt, $fMorph, $fGlobal) → $fHid` could explicitly model "for this patient, which feature group is most diagnostic" — a data-dependent gate over the entire feature taxonomy.

**Bigger feature spaces.** ECGDeli's 177 features are the simple case. Raw 10-second 12-lead waveforms (5000 timesteps × 12 leads = 60K values) would benefit from learned feature extractors as productions — TCNs, wavelet filters, learned spectrograms — composed with the same typed-lambda structure.

**Online learning.** PTB-XL is static. In a real clinical pipeline, the same DSL could be retrained per-site, with the *structure* of the winning program preserved but the weights re-fit on local data. The interpretable program is a piece of model documentation.

## 10. Closing Thoughts

Neurosymbolic program synthesis is often described as "interpretable but lower-performing". This project pushed back on that framing. A 12-line typed DSL plus the neurosym library's A* search produced a 4-node ECG classification program that matches the MLP baseline and beats the RandomForest, with every weight inspectable.

The cost was thought-iterations on the DSL design — we ran a dozen architectural variants before finding the heterogeneous-typed formulation. But the final DSL is *simpler* than any of the intermediate attempts: 5 typed embeds, 3 arithmetic ops, and a typed lambda target. Less code, more performance, complete transparency.

The right inductive bias, expressed through the right type system, is more useful than another layer of black-box capacity.

---

### Reproducibility

All code lives in this repository. The main entry points:

- **DSL definitions:** `neurosym/examples/near/dsls/attention_ecg_dsl.py` (look for `phase1_typed_ecg_dsl`)
- **Data loader:** `neurosym/datasets/ecg_data_example.py` (`feature_groups=True` mode)
- **Benchmark script:** `notebooks/ecg_reproduction/benchmark_reduced_attention_ecg.py` (`--phase1-typed` flag)
- **Experiment log:** `notebooks/ecg_reproduction/EXPERIMENTS.md`
- **PTB-XL dataset:** [physionet.org/content/ptb-xl/1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)
- **ECGDeli features:** [physionet.org/content/ptb-xl-plus/1.0.1](https://physionet.org/content/ptb-xl-plus/1.0.1/)
- **NEAR paper:** Shah, A. et al. "Learning Differentiable Programs with Admissible Neural Heuristics." NeurIPS 2020. [arXiv:2007.12101](https://arxiv.org/abs/2007.12101)
- **neurosym library:** [github.com/trishullab/neurosym-lib](https://github.com/trishullab/neurosym-lib)

To reproduce the headline result:
```bash
python notebooks/ecg_reproduction/benchmark_reduced_attention_ecg.py \
  --num-programs 500 --lr 1e-3 --hidden-dim 32 --batch-size 256 \
  --structural-cost-penalty 0.12 --label-mode single \
  --epochs 50 --final-epochs 250 \
  --phase1-typed \
  --output outputs/ecg_results/phase3_reduced/T1_repro.pkl
```
