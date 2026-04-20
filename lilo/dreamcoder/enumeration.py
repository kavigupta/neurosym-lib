import os
import re
import subprocess
import time as time
import traceback

import numpy as np
import torch
from dreamcoder.grammar import *
from dreamcoder.likelihoodModel import AllOrNothingLikelihoodModel
from dreamcoder.program import Program
from dreamcoder.tests.program_dist.utils import enumerate_dsl
from dreamcoder.utilities import get_root_dir, limit_virtual_memory_fn

import neurosym as ns
from neurosym.compression.process_abstraction import _StitchLambdaRewriter
from neurosym.dsl.abstraction import AbstractionProduction, _with_index_parameters
from neurosym.examples.dreamcoder.list_example import list_dsl, list_dslf
from neurosym.programs.s_expression_render import render_s_expression
from neurosym.types.type import ArrowType
from neurosym.types.type_signature import FunctionTypeSignature
from neurosym.types.type_string_repr import parse_type
from neurosym.types.type_with_environment import StrictEnvironment

DEFAULT_SOLVER_DIRECTORY = "."

INDUCTIVE_EXAMPLES_LIKELIHOOD_MODEL = "inductive_examples_likelihood_model"  # Only use the inductive examples to determine the likelihood.

INDUCTIVE_EXAMPLES_DISCOUNTED_PRIOR_LIKELIHOOD_MODEL = "inductive_examples_discounted_prior_likelihood_model"  # Use the inductive examples or otherwise use a discounted prior


class _TaskRootTypePreorderMaskELF(ns.TypePreorderMaskELF):
    """
    TypePreorderMaskELF variant that pins the root position to a single
    task-specific type rather than the DSL's ``valid_root_types``.

    The shared DSL is built with every root type the experiment may request
    so that its primitive set (and thus the bigram parameter tensor) is
    identical across tasks. At enumeration time, however, each task has
    exactly one request type; using the DSL's multi-valued
    ``valid_root_types`` would over-count valid productions at the root
    (and pick the wrong body type when multiple root types share a
    lambda's input signature), making NeuroSym log-probabilities diverge
    from DreamCoder's.
    """

    def __init__(self, tree_dist, dsl, task_root_type, dreamcoder_compat=False):
        # Must be set before super().__init__ so _root_type_for_entry can see it
        # if the base class ever consults it during construction.
        self._task_root_type = task_root_type
        super().__init__(tree_dist, dsl, dreamcoder_compat=dreamcoder_compat)

    def _validate_root_types(self):
        # The root type is supplied externally, so the DSL may legitimately
        # have any number of ``valid_root_types`` (we use those only for
        # primitive-set construction, not for choosing the root).
        assert (
            self.dsl.valid_root_types is not None
        ), "DSL must have valid_root_types set"

    def _root_type_for_entry(self):
        return self._task_root_type


def _infer_task_root_type(task):
    """Infer a NeuroSym root type for ``task`` from its first example shape.

    Mirrors the input/output inference the enumeration code used before a
    shared full DSL was introduced: we look at whether the example's input
    and output are lists of ints or ints, and construct the corresponding
    NeuroSym arrow type.
    """
    example = task.examples[0]
    input_type = "[i]" if isinstance(example[0][0], list) else "i"
    output_type = "[i]" if isinstance(example[1], list) else "i"
    return parse_type(f"{input_type} -> {output_type}")


def budgetIncrement(lb):
    if True:
        return 1.5
    # Very heuristic - not sure what to do here
    if lb < 24.0:
        return 1.0
    elif lb < 27.0:
        return 0.5
    else:
        return 0.25


import neurosym as ns
from neurosym.dsl.abstraction import _with_index_parameters


def parse_abstraction_dc_to_ns(
    abstraction: str, primitive_list: list[str] = None
) -> ns.InitializedSExpression | ns.AbstractionIndexParameter:
    """
    Convert a DreamCoder (Stitch) abstraction string into a NeuroSym
    ``InitializedSExpression`` / ``AbstractionIndexParameter`` tree.

    Example input:
        ``#(lambda (lambda (index $1 (sort $0))))``

    Tokenization and bracket-matching are delegated to
    ``ns.parse_s_expression``. The only wrinkle is Stitch's leading ``#``,
    which is not valid top-level s-expression syntax — we wrap the whole
    thing in outer parens so ``#`` becomes a regular head symbol, which
    the walker below recognizes as a stitch abstraction (body + applied
    args). Nested ``#(...)`` inside the body falls out of the recursion
    automatically because the parser already sees the surrounding parens.

    ``primitive_list`` is kept for backwards-compatibility with the
    previous signature but is no longer consulted — the parser derives
    structure from syntax alone.

    Raises ``ValueError`` on malformed input. DreamCoder abstractions
    should always be well-formed, so a failure here is a parser bug
    to be fixed rather than silently skipped.
    """
    del primitive_list
    src = f"({abstraction})" if abstraction.startswith("#") else abstraction
    return _convert_dc_sexp(ns.parse_s_expression(src, for_stitch=True))


def _convert_dc_sexp(
    s_exp,
) -> ns.InitializedSExpression | ns.AbstractionIndexParameter:
    if isinstance(s_exp, ns.SExpression):
        # Stitch abstraction: first child is the body, remaining are args
        # applied at this position. With no applied args the abstraction
        # just is its body.
        if s_exp.symbol == "#":
            body_exp, *arg_exps = s_exp.children
            # Peel outer ``lambda`` wrappers: in stitch, each ``lambda``
            # declares one abstraction parameter (bound by $0, $1, ...); the
            # NeuroSym representation encodes these via AbstractionIndexParameter
            # inside the body rather than as explicit lambda nodes.
            while (
                isinstance(body_exp, ns.SExpression)
                and body_exp.symbol == "lambda"
                and len(body_exp.children) == 1
            ):
                body_exp = body_exp.children[0]
            body = _convert_dc_sexp(body_exp)
            if not arg_exps:
                return body
            assert isinstance(body, ns.InitializedSExpression)
            arg_tuple = tuple(_convert_dc_sexp(a) for a in arg_exps)
            return _with_index_parameters(body, arg_tuple)
        # With ``for_stitch=True`` de Bruijn variables are wrapped as
        # empty-children ``SExpression('$N', ())`` rather than bare strings.
        # If applied to arguments (e.g. ``($1 0)``), the NeuroSym abstraction
        # representation cannot express higher-order variable application, so
        # we drop the applied args (matching the behavior in
        # tests/s_exp/dreamcoder_abstractions_test.py::parse_abstraction_dc_to_ns).
        if isinstance(s_exp.symbol, str) and s_exp.symbol.startswith("$"):
            return ns.AbstractionIndexParameter(int(s_exp.symbol[1:]))
        children = tuple(_convert_dc_sexp(c) for c in s_exp.children)
        return ns.InitializedSExpression(s_exp.symbol, children, {})
    assert isinstance(s_exp, str), f"Unexpected parser output: {s_exp!r}"
    if s_exp.startswith("$"):
        return ns.AbstractionIndexParameter(int(s_exp[1:]))
    return ns.InitializedSExpression(s_exp, (), {})


def neurosym_to_dreamcoder(s: str):
    # Convert NeuroSym binder tags like `lam_7` into DreamCoder's `lambda`.
    # We only rewrite full lambda-tag tokens, not arbitrary "lam" substrings.
    s = re.sub(r"\blam(?:_\d+)?\b", "lambda", s)
    # Collapse typed variable tags (e.g. $0_1) to DreamCoder de Bruijn indices.
    return re.sub(r"\$([0-9]+)_([0-9]+)", r"$\1", s)


def multicoreEnumeration(
    g,
    tasks,
    _=None,
    enumerationTimeout=None,
    solver="ocaml",
    CPUs=1,
    maximumFrontier=None,
    verbose=True,
    evaluationTimeout=None,
    testing=False,
    unigramGrammar=None,
    max_mem_per_enumeration_thread=1000000,
    solver_directory=DEFAULT_SOLVER_DIRECTORY,
    likelihood_model_string=INDUCTIVE_EXAMPLES_LIKELIHOOD_MODEL,
):
    """g: Either a Grammar, or a map from task to grammar.
    Returns (list-of-frontiers, map-from-task-to-search-time)"""

    assert (
        enumerationTimeout is not None
    ), "enumerateForTasks: You must provide a timeout."

    if not isinstance(g, dict):
        g = {t: g for t in tasks}

    if len(tasks) > 0:
        t = tasks[0]
        if isinstance(g[t], ContextualGrammar):
            print(f"Contextual Grammar Productions: {g[t].productions} \n")
            print(f"Contextual Grammar Primitives: {g[t].primitives} \n")
            # For contextual grammars, we need both the per-parent argument grammars (library)
            # and the special top-level grammar (noParent). DreamCoder treats lambdas as
            # structure (they do not become a "parent"), but NeuroSym will emit explicit
            # `lam_*` nodes; we therefore map `lam_*` parent distributions to the same
            # distribution DreamCoder would use at that position (noParent).
            library_list = {tx: g[tx].library for tx in tasks}
            no_parent_by_task = {tx: g[tx].noParent for tx in tasks}
            variable_parent_by_task = {tx: g[tx].variableParent for tx in tasks}
        elif isinstance(g[t], Grammar):
            task2grammar = g
            print(f"Expressions2Likelihood: {task2grammar[t].expression2likelihood}")
            library_list = {tx: task2grammar[tx].expression2likelihood for tx in tasks}

    dsl_dict = {}
    min_likelihood_dict = {task: 0.0 for task in tasks}
    enumerations = {task: [] for task in tasks}
    frontiers = {}
    dist_dict = {}

    # Must match the types used in listPrimitives.primitives() so the DSL
    # symbol names are identical to those in the DreamCoder grammar.
    from dreamcoder.domains.list.listPrimitives import LIST_OUTPUT_TYPES

    all_output_types = sorted(LIST_OUTPUT_TYPES)
    print(f"All DSL output types: {all_output_types}")

    for task in g.keys():
        dslf = list_dslf(*all_output_types)
        dsl = dslf.finalize()
        dsl_dict[task] = dsl
        # Each task has one specific request type; pin the root mask to it so
        # NeuroSym log-probs match DreamCoder's type-aware normalization.
        task_root_type = _infer_task_root_type(task)

        primitive_list = [prod.symbol() for prod in dsl.productions]

        for k, _val in library_list[task].items():
            if str(k) not in primitive_list and str(k)[0] != "$":
                # This means that k is likely to be an abstraction, so we generate an abstraction production from the DreamCoder abstraction as described in the grammar
                print(f"Adding abstraction {k} to neurosym DSL.")
                parsed_abstraction = parse_abstraction_dc_to_ns(str(k), primitive_list)
                s_exp = ns.SExpression(k, [parsed_abstraction])
                type_argument = [dsl.compute_type_abs(x) for x in s_exp.children][0]
                if type_argument is None:
                    print(
                        f"Skipping abstraction {k}: could not infer abstraction type."
                    )
                    continue

                if isinstance(type_argument.typ, ArrowType):
                    type_signature = FunctionTypeSignature.from_type(
                        type_argument.typ
                    )
                else:
                    # Build a function type from the inferred environment. Environment
                    # index 0 is the most recent binder, so we order arguments by
                    # descending environment index (outermost to innermost).
                    assert isinstance(
                        type_argument.env, StrictEnvironment
                    ), f"Expected a StrictEnvironment for abstraction {k}, but got {type(type_argument.env)}"
                    env_elements = type_argument.env._elements
                    argument_types = []
                    for env_index in sorted(env_elements.keys(), reverse=True):
                        env_typ = env_elements[env_index]
                        if isinstance(env_typ, tuple):
                            if len(env_typ) != 1:
                                raise ValueError(
                                    f"Unexpected env type tuple at index {env_index}: {env_typ}"
                                )
                            env_typ = env_typ[0]
                        argument_types.append(env_typ)
                    type_signature = FunctionTypeSignature(
                        argument_types,
                        type_argument.typ,
                    )

                print(
                    f"Parsed abstraction {k} to {s_exp} with type signature {type_signature}"
                )
                final = AbstractionProduction(str(k), type_signature, s_exp)
                dsl = dsl.add_productions(final)
                primitive_list.append(str(k))

        max_arity = 3  # for toy 1
        num_productions = 1 + len(dsl.productions)

        # print(f"Primitive List: {[f'{(x.symbol(), x.type_signature())} ---' for x in dsl.productions]} \n")
        # print(f"Primitive List: {[str(x.symbol()) + '----' for x in dsl.productions]} \n")

        # Now that we have the neurosym-equivalent DSL, we can create the BigramProgramDistributionFamily
        # family = ns.BigramProgramDistributionFamily(dsl)
        family = ns.BigramProgramDistributionFamily(
            dsl,
            include_type_preorder_mask=False,
            additional_preorder_masks=[
                lambda td, dsl, trt=task_root_type: _TaskRootTypePreorderMaskELF(
                    td, dsl, trt, dreamcoder_compat=True
                ),
            ],
            dreamcoder_compat=True,
        )
        print(f"Bigram Parameters Shape is: {family.parameters_shape()}")

        ordered_symbols = dsl.ordered_symbols(include_root=True)
        dreamcoder_ns_mapping = {}
        for p in range(len(ordered_symbols)):
            dreamcoder_ns_mapping[ordered_symbols[p]] = p
        print(f"DreamCoder NeuroSym Mapping: {list(dreamcoder_ns_mapping)} \n")

        # Use float64 end-to-end here to minimize numerical drift between
        # DreamCoder scoring and NeuroSym likelihood computations.
        # IMPORTANT: this tensor is in log-space. Unset entries must be -inf
        # (impossible), not 0.0 (log(1.0)), or invalid transitions leak in.
        dist = np.full(
            (num_productions, max_arity, num_productions),
            -np.inf,
            dtype=np.float64,
        )
        print("likelihood dict!")

        # Automatically maps DreamCoder primitive name to one or more NeuroSym indices.
        def _child_to_inds(child):
            """
            Map a DreamCoder child symbol to the corresponding NeuroSym symbol
            index/indices.

            For most primitives this is just a string match. For the special
            variable symbol "$0" we want the *same* log-score to be attached to
            *all* concrete NeuroSym variable symbols ("$N_M").  DreamCoder uses
            a single "$0" to represent any variable and adjusts with
            numberOfFreeVariables; NeuroSym has separate symbols per de Bruijn
            index and type variant, with dreamcoder_compat handling the split.
            """
            s = str(child)
            if s == "$0":
                # Attach the same log-score to variales.
                # The type preorder mask with dreamcoder_compat will
                # filter to only the type-valid variables at each position and
                # divide by n_vars.
                # NeuroSym historically emitted variables as "$N_M" (N = de
                # Bruijn index, M = type variant). After the main-branch merge
                # that allowed parameterized output types, type suffixes were
                # dropped, so the current form is just "$N". Match both.
                var_syms = [
                    sym
                    for sym in ordered_symbols
                    if isinstance(sym, str) and re.fullmatch(r"\$\d+(_\d+)?", sym)
                ]
                assert (
                    len(var_syms) > 0
                ), 'Expected at least one "$N" variable symbol in NeuroSym DSL.'
                return [dreamcoder_ns_mapping[sym] for sym in var_syms]
            assert s in dreamcoder_ns_mapping, (
                f"DreamCoder grammar references symbol {s!r} that is absent "
                "from the NeuroSym DSL. This usually means an abstraction was "
                "skipped above (compute_type_abs returned None) or a primitive "
                "in the DC grammar is missing from list_dslf — fix that rather "
                "than silently dropping the symbol."
            )
            return [dreamcoder_ns_mapping[s]]

        if isinstance(g[task], ContextualGrammar):
            # 1) Fill the root distribution using DreamCoder's noParent grammar.
            root_ind = dreamcoder_ns_mapping["<root>"]
            root_row = no_parent_by_task[task].expression2likelihood
            for child, likelihood in root_row.items():
                child_inds = _child_to_inds(child)
                # Root has no argument-index notion in DreamCoder, but NeuroSym's bigram
                # model expects an arity axis; use the same distribution for all arities.
                for arity in range(max_arity):
                    for child_ind in child_inds:
                        dist[root_ind][arity][child_ind] = likelihood

            # 2) Fill all lambda-parent distributions from noParent as well.
            lam_parent_inds = [
                ind
                for prod, ind in dreamcoder_ns_mapping.items()
                if str(prod).startswith("lam_")
            ]
            for lam_ind in lam_parent_inds:
                for child, likelihood in root_row.items():
                    child_inds = _child_to_inds(child)
                    # Mirror root behavior across arity slots to avoid leaving
                    # unset lambda rows that become uniform after masking.
                    for arity in range(max_arity):
                        for child_ind in child_inds:
                            dist[lam_ind][arity][child_ind] = likelihood

            # 3) Fill each parent’s per-argument distributions from the library.
            likelihood_dict = library_list[
                task
            ]  # {parent_expr: [Grammar_for_arg0, ...]}
            for parent, arg_grammars in likelihood_dict.items():
                parent_key = str(parent)
                assert parent_key in dreamcoder_ns_mapping, (
                    f"Contextual-grammar parent {parent_key!r} is absent from "
                    "the NeuroSym DSL. See _child_to_inds for the likely cause."
                )
                parent_ind = dreamcoder_ns_mapping[parent_key]
                # If for some reason we have no arg grammars, leave it to normalization.
                for arg_index, arg_grammar in enumerate(arg_grammars[:max_arity]):
                    bigram_row = arg_grammar.expression2likelihood
                    for child, likelihood in bigram_row.items():
                        child_inds = _child_to_inds(child)
                        for child_ind in child_inds:
                            dist[parent_ind][arg_index][child_ind] = likelihood

            # 4) Variable-parent distributions (DreamCoder uses this when parent is an Index).
            # NeuroSym doesn't have an explicit "Index parent" node; we approximate by
            # also applying this distribution to lambda parents (handled above) and root.
            # If later we introduce an explicit node for "variable parent", wire it here.
            _ = variable_parent_by_task  # currently unused beyond storing for future fixes

        elif isinstance(g[task], Grammar):
            likelihood_dict = library_list[task]  # expression2likelihood
            for k, v in likelihood_dict.items():
                production_inds = _child_to_inds(k)
                for production_ind in production_inds:
                    for prod_ind in range(num_productions):
                        for arity in range(max_arity):
                            dist[prod_ind][arity][production_ind] = v
            # Lambda is structural in DreamCoder (absent from expression2likelihood).
            # Set lam_* child slots to log-prob 0 (prob 1) so they remain reachable;
            # TypePreorderMaskELF will allow them only where the expected type is ArrowType.
            lam_child_inds = [
                ind
                for sym, ind in dreamcoder_ns_mapping.items()
                if str(sym).startswith("lam_")
            ]
            for lam_child_ind in lam_child_inds:
                for prod_ind in range(num_productions):
                    for arity in range(max_arity):
                        dist[prod_ind][arity][lam_child_ind] = 0.0

        # Filter the productions out that are impossible to reach
        tensor_dist = torch.tensor(dist)
        reshaped_tensor_dist = np.exp(
            tensor_dist.reshape(tuple([1] + list(tensor_dist.shape)))
        )
        # Apply only the validity mask; keep DreamCoder's scores as logits.
        # masked_logits = family._normalize_parameters(reshaped_tensor_dist)
        dist_dict[task] = ns.BigramProgramDistribution(
            dist_fam=family,
            distribution=reshaped_tensor_dist.numpy()[0],
            _disable_validation=True,
        )
        StitchRewriter = _StitchLambdaRewriter(dsl_dict[task])

        # SAGNIK_TBD: Recreate the while loop below - while satisfying memory and time constraints, enumerate all programs in between the lower and upper bound
        # SAGNIK_TBD: Optimize for one CPU per job
        starting = time.time()
        parsed_enumerations = []
        solved = False
        while not solved:
            bi = budgetIncrement(min_likelihood_dict[task])
            current_generations = list(
                family.enumerate(
                    dist=dist_dict[task],
                    min_likelihood=min_likelihood_dict[task],
                    chunk_size=bi,
                )
            )
            enumerations[task] += current_generations

            # Here we attempt to parse enumerated programs to DreamCoder-style programs to create DreamCoder Frontier Entries
            for ns_prog_tuple in current_generations:
                ns_prog, prob_fraction = ns_prog_tuple
                try:
                    actual_ns_function = dsl.compute(dsl.initialize(ns_prog))
                except Exception as e:
                    # print(f"Exception {e} for {render_s_expression(ns_prog)}")
                    continue
                target_outputs = []
                actual_outputs = []
                for example in task.examples:
                    target_outputs.append(example[1])
                    try:
                        actual_outputs.append(actual_ns_function(*example[0]))
                    except Exception as e:
                        actual_outputs.append(None)
                likelihood = 0.0
                for actual, target in zip(actual_outputs, target_outputs):
                    if actual == None or actual != target:
                        likelihood = float("-inf")
                        break
                if likelihood == 0.0:
                    solved = True
                try:
                    dreamcoder_prog = Program.parse(
                        neurosym_to_dreamcoder(
                            render_s_expression(StitchRewriter.to_stitch(ns_prog))
                        )
                    )
                except Exception as e:
                    print(f"Grammar is {g[task]}")
                    print(f"Exception {e} for {render_s_expression(ns_prog)}")
                    raise e
                log_prob = float(format(prob_fraction, ".10g"))
                dreamcoder_entry = FrontierEntry(
                    program=dreamcoder_prog, logPrior=log_prob, logLikelihood=likelihood
                )
                rescored_log_prob = g[task].logLikelihood(task.request, dreamcoder_prog)
                prob_fraction_2 = family.compute_likelihood(dist_dict[task], ns_prog)
                assert (
                    abs(prob_fraction - prob_fraction_2) < 1e-5
                ), f"Probability mismatch within NeuroSym!: {prob_fraction} [log({np.exp(prob_fraction)})] vs {prob_fraction_2} [log({np.exp(prob_fraction_2)})] in {render_s_expression(ns_prog)}"
                assert (
                    abs(rescored_log_prob - prob_fraction_2) < 1e-3
                ), f"Log probability mismatch between DreamCoder and NeuroSym!: {rescored_log_prob} [log({np.exp(rescored_log_prob)})] vs {prob_fraction_2} [log({np.exp(prob_fraction_2)})]  in {render_s_expression(ns_prog)}"
                parsed_enumerations.append(dreamcoder_entry)
            min_likelihood_dict[task] -= bi
            if time.time() - starting > enumerationTimeout:
                print(
                    f"Final min_likelihood for task {task} is {min_likelihood_dict[task]}, enumerated {len(parsed_enumerations)} programs."
                )
                break
        print(f"Task {task} done with solve status {solved}.")
        frontiers[task] = Frontier(parsed_enumerations, task=task)
    bestSearchTime = {t: None for t in tasks}
    return [frontiers[t] for t in tasks], bestSearchTime

    # use expression2likelihood for bigram in dsl
    enumerations = list(enumerate_dsl(family, dist, min_likelihood=-1000000))
    print(enumerations)
    # for t in tasks:
    #     frontier = Frontier(
    #         [
    #             FrontierEntry(
    #                 program = Program.parse(e[0]),
    #                 logLikelihood= e[1].as_integer_ratio()[0]/e[1].as_integer_ratio()[1],
    #                 #tokens= escape_tokens(e["tokens"]).split(),
    #                 logPrior= g.logLikelihood(t.request, Program.parse(e[0])),
    #             )
    #             for e in enumerations
    #         ],
    #         task=t,
    #     )

    # frontiers = {t: Frontier([Program.parse(i[0]) for i in enumerations], task=t) for t in task2grammar}
    frontiers = {t: Frontier([], task=t) for t in task2grammar}
    bestSearchTime = {t: None for t in task2grammar}

    # # We don't use actual threads but instead use the multiprocessing
    # # library. This is because we need to be able to kill workers.
    # # from multiprocess import Process, Queue
    # print(
    #     f"Beginning enumeration on a total of {len(tasks)} tasks with a total of {CPUs} CPUs."
    # )
    # from multiprocessing import Queue

    # # everything that gets sent between processes will be dilled
    # import dill

    # solvers = {
    #     "ocaml": solveForTask_ocaml,
    #     "pypy": solveForTask_pypy,
    #     "python": solveForTask_python,
    # }
    # assert (
    #     solver in solvers
    # ), "You must specify a valid solver. options are ocaml, pypy, or python."

    # likelihoodModel = None
    # if solver == "pypy" or solver == "python":
    #     # Use an all or nothing likelihood model.
    #     likelihoodModel = AllOrNothingLikelihoodModel(timeout=evaluationTimeout)
    # if solver == "ocaml":
    #     likelihoodModel = likelihood_model_string

    # solver = solvers[solver]

    # if not isinstance(g, dict):
    #     g = {t: g for t in tasks}
    # task2grammar = g

    # # If we are not evaluating on held out testing tasks:
    # # Bin the tasks by request type and grammar
    # # If these are the same then we can enumerate for multiple tasks simultaneously
    # # If we are evaluating testing tasks:
    # # Make sure that each job corresponds to exactly one task
    # jobs = {}
    # for i, t in enumerate(tasks):
    #     if testing:
    #         k = (task2grammar[t], t.request, i)
    #     else:
    #         k = (task2grammar[t], t.request)
    #     jobs[k] = jobs.get(k, []) + [t]

    # disableParallelism = len(jobs) == 1
    # parallelCallback = (
    #     launchParallelProcess
    #     if not disableParallelism
    #     else lambda f, *a, **k: f(*a, **k)
    # )
    # if disableParallelism:
    #     eprint("Disabling parallelism on the Python side because we only have one job.")
    #     eprint("If you are using ocaml, there could still be parallelism.")

    # # Map from task to the shortest time to find a program solving it
    # bestSearchTime = {t: None for t in task2grammar}

    # lowerBounds = {k: 0.0 for k in jobs}

    # frontiers = {t: Frontier([], task=t) for t in task2grammar}

    # # For each job we keep track of how long we have been working on it
    # stopwatches = {t: Stopwatch() for t in jobs}

    # # Map from task to how many programs we enumerated for that task
    # taskToNumberOfPrograms = {t: 0 for t in tasks}

    # def numberOfHits(f):
    #     return sum(e.logLikelihood > -0.01 for e in f)

    # def budgetIncrement(lb):
    #     if True:
    #         return 1.5
    #     # Very heuristic - not sure what to do here
    #     if lb < 24.0:
    #         return 1.0
    #     elif lb < 27.0:
    #         return 0.5
    #     else:
    #         return 0.25

    # def maximumFrontiers(j):
    #     tasks = jobs[j]
    #     return {t: maximumFrontier - numberOfHits(frontiers[t]) for t in tasks}

    # def allocateCPUs(n, tasks):
    #     allocation = {t: 0 for t in tasks}
    #     while n > 0:
    #         for t in tasks:
    #             # During testing we use exactly one CPU per task
    #             if testing and allocation[t] > 0:
    #                 return allocation
    #             allocation[t] += 1
    #             n -= 1
    #             if n == 0:
    #                 break
    #     return allocation

    # def refreshJobs():
    #     for k in list(jobs.keys()):
    #         v = [
    #             t
    #             for t in jobs[k]
    #             if numberOfHits(frontiers[t]) < maximumFrontier
    #             and stopwatches[k].elapsed <= enumerationTimeout
    #         ]
    #         if v:
    #             jobs[k] = v
    #         else:
    #             del jobs[k]

    # # Workers put their messages in here
    # q = Queue()

    # # How many CPUs are we using?
    # activeCPUs = 0

    # # How many CPUs was each job allocated?
    # id2CPUs = {}
    # # What job was each ID working on?
    # id2job = {}
    # nextID = 0

    # while True:
    #     refreshJobs()
    #     # Don't launch a job that we are already working on
    #     # We run the stopwatch whenever the job is being worked on
    #     # freeJobs are things that we are not working on but could be
    #     freeJobs = [
    #         j
    #         for j in jobs
    #         if not stopwatches[j].running
    #         and stopwatches[j].elapsed < enumerationTimeout - 0.5
    #     ]
    #     if freeJobs and activeCPUs < CPUs:
    #         # Allocate a CPU to each of the jobs that we have made the least
    #         # progress on
    #         freeJobs.sort(key=lambda j: lowerBounds[j])
    #         # Launch some more jobs until all of the CPUs are being used
    #         availableCPUs = CPUs - activeCPUs
    #         allocation = allocateCPUs(availableCPUs, freeJobs)
    #         for j in freeJobs:
    #             if allocation[j] == 0:
    #                 continue
    #             g, request = j[:2]
    #             bi = budgetIncrement(lowerBounds[j])
    #             thisTimeout = enumerationTimeout - stopwatches[j].elapsed
    #             eprint(
    #                 "(python) Launching %s (%d tasks) w/ %d CPUs. %f <= MDL < %f. Timeout %f."
    #                 % (
    #                     request,
    #                     len(jobs[j]),
    #                     allocation[j],
    #                     lowerBounds[j],
    #                     lowerBounds[j] + bi,
    #                     thisTimeout,
    #                 )
    #             )
    #             stopwatches[j].start()
    #             parallelCallback(
    #                 wrapInThread(solver),
    #                 q=q,
    #                 g=g,
    #                 ID=nextID,
    #                 elapsedTime=stopwatches[j].elapsed,
    #                 CPUs=allocation[j],
    #                 tasks=jobs[j],
    #                 lowerBound=lowerBounds[j],
    #                 upperBound=lowerBounds[j] + bi,
    #                 budgetIncrement=bi,
    #                 timeout=thisTimeout,
    #                 evaluationTimeout=evaluationTimeout,
    #                 maximumFrontiers=maximumFrontiers(j),
    #                 testing=testing,
    #                 likelihoodModel=likelihoodModel,
    #                 unigramGrammar=unigramGrammar,
    #                 max_mem_per_enumeration_thread=max_mem_per_enumeration_thread,
    #                 solver_directory=solver_directory,
    #             )
    #             id2CPUs[nextID] = allocation[j]
    #             id2job[nextID] = j
    #             nextID += 1

    #             activeCPUs += allocation[j]
    #             lowerBounds[j] += bi

    #     # If nothing is running, and we just tried to launch jobs,
    #     # then that means we are finished
    #     if all(not s.running for s in stopwatches.values()):
    #         break

    #     # Wait to get a response
    #     message = Bunch(dill.loads(q.get()))

    #     if message.result == "failure":
    #         eprint("PANIC! Exception in child worker:", message.exception)
    #         eprint(message.stacktrace)
    #         assert False
    #     elif message.result == "success":
    #         # Mark the CPUs is no longer being used and pause the stopwatch
    #         activeCPUs -= id2CPUs[message.ID]
    #         stopwatches[id2job[message.ID]].stop()

    #         newFrontiers, searchTimes, pc = message.value
    #         for t, f in newFrontiers.items():
    #             oldBest = None if len(frontiers[t]) == 0 else frontiers[t].bestPosterior
    #             frontiers[t] = frontiers[t].combine(f)
    #             newBest = None if len(frontiers[t]) == 0 else frontiers[t].bestPosterior

    #             taskToNumberOfPrograms[t] += pc

    #             dt = searchTimes[t]
    #             if dt is not None:
    #                 if bestSearchTime[t] is None:
    #                     bestSearchTime[t] = dt
    #                 else:
    #                     # newBest & oldBest should both be defined
    #                     assert oldBest is not None
    #                     assert newBest is not None
    #                     newScore = newBest.logPrior + newBest.logLikelihood
    #                     oldScore = oldBest.logPrior + oldBest.logLikelihood

    #                     if newScore > oldScore:
    #                         bestSearchTime[t] = dt
    #                     elif newScore == oldScore:
    #                         bestSearchTime[t] = min(bestSearchTime[t], dt)
    #     else:
    #         eprint("Unknown message result:", message.result)
    #         assert False

    # eprint(
    #     "We enumerated this many programs, for each task:\n\t",
    #     list(taskToNumberOfPrograms.values()),
    # )

    # return [frontiers[t] for t in tasks], bestSearchTime


def wrapInThread(f):
    """
    Returns a function that is designed to be run in a thread/threadlike process.
    Result will be either put into the q
    """
    import dill

    def _f(*a, **k):
        q = k.pop("q")
        ID = k.pop("ID")

        try:
            r = f(*a, **k)
            q.put(dill.dumps({"result": "success", "ID": ID, "value": r}))
        except Exception as e:
            q.put(
                dill.dumps(
                    {
                        "result": "failure",
                        "exception": e,
                        "stacktrace": traceback.format_exc(),
                        "ID": ID,
                    }
                )
            )
            return

    return _f


OCAML_TEST_FLAG = "is_ocaml_test"  # Indicates a JSON response intended for testing.


def solveForTask_ocaml(
    _=None,
    elapsedTime=0.0,
    CPUs=1,
    g=None,
    tasks=None,
    lowerBound=None,
    upperBound=None,
    budgetIncrement=None,
    timeout=None,
    testing=None,  # FIXME: unused
    likelihoodModel=None,
    evaluationTimeout=None,
    maximumFrontiers=None,
    unigramGrammar=None,
    verbose=False,
    max_mem_per_enumeration_thread=1000000,
    solver_directory=DEFAULT_SOLVER_DIRECTORY,
):

    import json

    def taskMessage(t):
        serialized_examples = []
        for xs, y in t.examples:
            if hasattr(t, "serializeSpecialInput"):
                xs = t.serializeSpecialInput(xs)
            if hasattr(t, "serializeSpecialOutput"):
                y = t.serializeSpecialOutput(y, is_output=True)
            serialized_examples.append({"inputs": list(xs), "output": y})

        m = {
            "examples": serialized_examples,
            "name": t.name,
            "request": t.request.json(),
            "maximumFrontier": maximumFrontiers[t],
        }
        if hasattr(t, "specialTask"):
            special, extra = t.specialTask
            m["specialTask"] = special
            m["extras"] = extra
        if hasattr(t, "raw_programs_to_test"):
            m["raw_programs_to_test"] = t.raw_programs_to_test
        return m

    message = {
        "DSL": g.json(),
        "tasks": [taskMessage(t) for t in tasks],
        "programTimeout": float(evaluationTimeout),
        "nc": CPUs,
        "timeout": timeout,
        "lowerBound": lowerBound,
        "upperBound": upperBound,
        "budgetIncrement": budgetIncrement,
        "verbose": verbose,
        "shatter": 5 if len(tasks) == 1 and "turtle" in str(tasks[0].request) else 10,
        "likelihoodModel": likelihoodModel,
    }

    if hasattr(tasks[0], "maxParameters") and tasks[0].maxParameters is not None:
        message["maxParameters"] = tasks[0].maxParameters

    message = json.dumps(message)
    # uncomment this if you want to save the messages being sent to the solver

    solver_file = "solver"
    if hasattr(tasks[0], "specialSolver"):
        solver_file = tasks[0].specialSolver

    try:
        solver_file = os.path.join(get_root_dir(), solver_directory, solver_file)
        process = subprocess.Popen(
            solver_file, stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        limit_virtual_memory_with_psutil_if_possible(
            process, max_mem_per_enumeration_thread
        )

        response, error = process.communicate(bytes(message, encoding="utf-8"))
        response = json.loads(response.decode("utf-8"))
    except OSError as exc:
        raise exc

    except:
        print("response:", response)
        print("error:", error)
        with open("message", "w") as f:
            f.write(message)
        # Don't fail on errors
        # assert False, "MAX RAISE"
        print(
            "ERROR in enumeration, returning empty frontiers for this batch of tasks."
        )
        response = {t.name: [] for t in tasks}  # Empty response

    def escape_tokens(tokens):
        if unigramGrammar is not None:
            return unigramGrammar.escape_tokens_string(tokens)
        return g.escape_tokens_string(tokens)

    if OCAML_TEST_FLAG in response:
        return response

    pc = response.get("number_enumerated", 0)  # TODO
    frontiers = {}
    searchTimes = {}
    for t in tasks:
        solutions = response[t.name]
        frontier = Frontier(
            [
                FrontierEntry(
                    program=p,
                    logLikelihood=e["logLikelihood"],
                    tokens=escape_tokens(e["tokens"]).split(),
                    logPrior=g.logLikelihood(t.request, p),
                )
                for e in solutions
                for p in [Program.parse(e["program"])]
            ],
            task=t,
        )
        frontiers[t] = frontier
        if frontier.empty:
            searchTimes[t] = None
        # This is subtle:
        # The search time we report is actually not be minimum time to find any solution
        # Rather it is the time to find the MAP solution
        # This is important for regression problems,
        # where we might find something with a good prior but bad likelihood early on,
        # and only later discovered the good high likelihood program
        else:
            searchTimes[t] = (
                min((e["logLikelihood"] + e["logPrior"], e["time"]) for e in solutions)[
                    1
                ]
                + elapsedTime
            )

    return frontiers, searchTimes, pc


def solveForTask_pypy(
    _=None,
    elapsedTime=0.0,
    g=None,
    task=None,
    lowerBound=None,
    upperBound=None,
    budgetIncrement=None,
    timeout=None,
    likelihoodModel=None,
    evaluationTimeout=None,
    maximumFrontier=None,
    testing=False,
    unigramGrammar=None,
):
    return callCompiled(
        enumerateForTasks,
        g,
        tasks,
        likelihoodModel,
        timeout=timeout,
        testing=testing,
        elapsedTime=elapsedTime,
        evaluationTimeout=evaluationTimeout,
        maximumFrontiers=maximumFrontiers,
        budgetIncrement=budgetIncrement,
        lowerBound=lowerBound,
        upperBound=upperBound,
        unigramGrammar=None,
    )


def solveForTask_python(
    _=None,
    elapsedTime=0.0,
    g=None,
    tasks=None,
    lowerBound=None,
    upperBound=None,
    budgetIncrement=None,
    timeout=None,
    CPUs=1,
    likelihoodModel=None,
    evaluationTimeout=None,
    maximumFrontiers=None,
    testing=False,
    unigramGrammar=None,
):
    return enumerateForTasks(
        g,
        tasks,
        likelihoodModel,
        timeout=timeout,
        testing=testing,
        elapsedTime=elapsedTime,
        evaluationTimeout=evaluationTimeout,
        maximumFrontiers=maximumFrontiers,
        budgetIncrement=budgetIncrement,
        lowerBound=lowerBound,
        upperBound=upperBound,
        unigramGrammar=None,
    )


class EnumerationTimeout(Exception):
    pass


def enumerateForTasks(
    g,
    tasks,
    likelihoodModel,
    _=None,
    verbose=False,
    timeout=None,
    elapsedTime=0.0,
    CPUs=1,
    testing=False,  # unused
    evaluationTimeout=None,
    lowerBound=0.0,
    upperBound=100.0,
    budgetIncrement=1.0,
    maximumFrontiers=None,
    unigramGrammar=None,
):
    assert timeout is not None, "enumerateForTasks: You must provide a timeout."

    from time import time

    request = tasks[0].request
    assert all(
        t.request == request for t in tasks
    ), "enumerateForTasks: Expected tasks to all have the same type"

    maximumFrontiers = [maximumFrontiers[t] for t in tasks]
    # store all of the hits in a priority queue
    # we will never maintain maximumFrontier best solutions
    hits = [PQ() for _ in tasks]

    starting = time()
    previousBudget = lowerBound
    budget = lowerBound + budgetIncrement
    try:
        totalNumberOfPrograms = 0
        while (
            time() < starting + timeout
            and any(len(h) < mf for h, mf in zip(hits, maximumFrontiers))
            and budget <= upperBound
        ):
            numberOfPrograms = 0

            for prior, _, p in g.enumeration(
                Context.EMPTY,
                [],
                request,
                maximumDepth=99,
                upperBound=budget,
                lowerBound=previousBudget,
            ):
                descriptionLength = -prior
                # Shouldn't see it on this iteration
                assert descriptionLength <= budget
                # Should already have seen it
                assert descriptionLength > previousBudget

                numberOfPrograms += 1
                totalNumberOfPrograms += 1

                for n in range(len(tasks)):
                    task = tasks[n]

                    # Warning:changed to max's new likelihood model situation
                    # likelihood = task.logLikelihood(p, evaluationTimeout)
                    # if invalid(likelihood):
                    # continue
                    success, likelihood = likelihoodModel.score(p, task)
                    if not success:
                        continue

                    dt = time() - starting + elapsedTime
                    priority = -(likelihood + prior)
                    hits[n].push(
                        priority,
                        (
                            dt,
                            FrontierEntry(
                                program=p, logLikelihood=likelihood, logPrior=prior
                            ),
                        ),
                    )
                    if len(hits[n]) > maximumFrontiers[n]:
                        hits[n].popMaximum()

                if timeout is not None and time() - starting > timeout:
                    raise EnumerationTimeout

            previousBudget = budget
            budget += budgetIncrement

            if budget > upperBound:
                break
    except EnumerationTimeout:
        pass
    frontiers = {
        tasks[n]: Frontier([e for _, e in hits[n]], task=tasks[n])
        for n in range(len(tasks))
    }
    searchTimes = {
        tasks[n]: None if len(hits[n]) == 0 else min(t for t, _ in hits[n])
        for n in range(len(tasks))
    }

    return frontiers, searchTimes, totalNumberOfPrograms
