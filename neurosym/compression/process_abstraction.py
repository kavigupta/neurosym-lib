import collections
import itertools
import os
import sys
import uuid
from typing import List, Union

import stitch_core

from neurosym.dsl.dsl import DSL

from ..dsl.abstraction import AbstractionIndexParameter, AbstractionProduction
from ..programs.s_expression import (
    InitializedSExpression,
    SExpression,
    postorder,
)
from ..programs.s_expression_render import (
    parse_s_expression,
    render_s_expression,
    symbols_for_program,
)
from ..types.type import ArrowType
from ..types.type_signature import FunctionTypeSignature


def _compute_abstraction_production(
    dsl,
    s_expression_using: SExpression,
    abstr_name: str,
    abstr_body: SExpression,
):
    """
    Compute the type of an abstraction production.

    Args:
        dsl: The DSL.
        s_expression_using: An SExpression using the abstraction.
        abstr: The original abstraction.

    Returns an AbstractionProduction corresponding to the abstraction.
    """
    abstr_body = _inject_parameters(abstr_body)
    usage = next(x for x in postorder(s_expression_using) if x.symbol == abstr_name)
    type_arguments = [dsl.compute_type(x) for x in usage.children]
    type_out = dsl.compute_type(
        abstr_body,
        lambda x: (
            type_arguments[x.index]
            if isinstance(x, AbstractionIndexParameter)
            else None
        ),
    ).typ

    # Resolve type variables (e.g. from polymorphic $0 in usage children)
    # by matching each argument type against the corresponding subtree
    # in the original program. The original subtree is the child of the
    # usage position, expanded by the abstraction body.
    arg_types = [x.typ for x in type_arguments]
    if any(t.get_type_vars() for t in arg_types) or type_out.get_type_vars():
        mapping = _resolve_abstraction_vars(
            dsl, s_expression_using, abstr_name, abstr_body
        )
        arg_types = [t.subst_type_vars(mapping) for t in arg_types]
        type_out = type_out.subst_type_vars(mapping)
    type_signature = FunctionTypeSignature(arg_types, type_out)

    return AbstractionProduction(abstr_name, type_signature, abstr_body)


def _resolve_abstraction_vars(dsl, s_expression_using, abstr_name, abstr_body):
    """Resolve type variables in abstraction types by reconstructing the original program.

    Expands the abstraction body with usage children to reconstruct the original
    subtree, computes its type (fully resolved), then unifies the polymorphic
    argument types with the resolved types to get a var mapping.
    """
    usage = next(x for x in postorder(s_expression_using) if x.symbol == abstr_name)
    original = _expand_abstraction(abstr_body, usage.children)

    # Compute types for each child both with and without context.
    # The "without context" type has type variables; the "with context"
    # type (from the expanded tree) has resolved types.
    resolved_twe = dsl.compute_type(original)
    mapping = {}
    for child in usage.children:
        poly_twe = dsl.compute_type(child)
        if not poly_twe.typ.get_type_vars():
            continue
        for var_name in poly_twe.typ.get_type_vars():
            if not var_name.startswith("__var_"):
                continue
            idx = int(var_name[len("__var_") :])
            resolved_type = resolved_twe.env.type_at(idx)
            if resolved_type is not None:
                mapping[var_name] = resolved_type
    return mapping


def _expand_abstraction(abstr_body, children):
    """Expand an abstraction body by replacing AbstractionIndexParameters with children."""
    if isinstance(abstr_body, AbstractionIndexParameter):
        return children[abstr_body.index]
    return SExpression(
        abstr_body.symbol,
        tuple(_expand_abstraction(c, children) for c in abstr_body.children),
    )


def _inject_parameters(s_expression: Union[SExpression, str]):
    """
    Inject parameters into an SExpression, replacing leaves of the form #N with
    AbstractionIndexParameter(N).
    """
    if isinstance(s_expression, str):
        assert s_expression.startswith("#")
        return AbstractionIndexParameter(int(s_expression[1:]))
    return SExpression(
        s_expression.symbol,
        tuple(_inject_parameters(x) for x in s_expression.children),
    )


def _next_symbol(dsl):
    """
    Next free symbol of the form __N0 in the DSL.

    Returns __N, the prefix.
    """
    possible_conflict = [
        x[2:-1] for x in dsl.symbols() if x.startswith("__") and x.endswith("0")
    ]
    possible_conflict = {int(x) for x in possible_conflict if x.isnumeric()}
    number = next(x for x in itertools.count(1) if x not in possible_conflict)
    return f"__{number}"


def _multi_lambda_to_single_lambda(dsl):
    lams = [x for x in dsl.productions if x.base_symbol() == "lam"]
    if not lams:
        return 0, {}
    max_index = max(x.get_numerical_index() for x in lams) + 1
    multi_to_single = {}
    zero_arg_lambda = None
    for lam in lams:
        if lam.arity == 0:
            assert zero_arg_lambda is None
            zero_arg_lambda = lam.get_numerical_index()
        if lam.arity == 1:
            continue
        multi_to_single[lam.get_numerical_index()] = [
            max_index + i for i in range(lam.arity)
        ]
        max_index += lam.arity
    return zero_arg_lambda, multi_to_single


class _StitchLambdaRewriter:
    """
    Rewrites a DSL to use only single-argument lambdas.

    Zero-argument lambdas are sent to a special symbol.

    Multi-argument lambdas are rewritten to multi single-argument lambdas.
    """

    def __init__(self, dsl):
        self.dsl = dsl
        (
            self.zero_arg_lambda_index_original,
            self.multi_to_single,
        ) = _multi_lambda_to_single_lambda(dsl)
        self.zero_arg_lambda_symbol = uuid.uuid4().hex

        self.first_single_to_multi = {
            single[0]: multi for multi, single in self.multi_to_single.items()
        }

        self.fused_lambda_tags = ",".join(
            sorted(
                {str(tag) for tags in self.multi_to_single.values() for tag in tags[1:]}
            )
        )

    def to_stitch(self, s_exp):
        if isinstance(s_exp, str):
            return s_exp
        children = tuple(self.to_stitch(x) for x in s_exp.children)
        if not s_exp.symbol.startswith("lam"):
            return SExpression(
                s_exp.symbol,
                children,
            )
        prod = self.dsl.get_production(s_exp.symbol)
        if prod.arity == 0:
            return SExpression(
                self.zero_arg_lambda_symbol,
                children,
            )
        if prod.arity == 1:
            return SExpression(
                s_exp.symbol,
                children,
            )
        assert prod.arity > 1
        [result] = children
        for i in reversed(range(prod.arity)):
            result = SExpression(
                f"lam_{self.multi_to_single[prod.get_numerical_index()][i]}",
                (result,),
            )
        return result

    def from_stitch(self, s_exp):
        if isinstance(s_exp, str):
            return s_exp
        children = lambda: tuple(self.from_stitch(x) for x in s_exp.children)
        if s_exp.symbol == self.zero_arg_lambda_symbol:
            return SExpression(
                f"lam_{self.zero_arg_lambda_index_original}",
                children(),
            )
        if s_exp.symbol in self.dsl.symbols() or not s_exp.symbol.startswith("lam"):
            return SExpression(s_exp.symbol, children())
        index = int(s_exp.symbol[4:])
        original = self.first_single_to_multi[index]
        for i in range(len(self.multi_to_single[original])):
            expected = f"lam_{self.multi_to_single[original][i]}"
            assert s_exp.symbol == expected, f"{s_exp.symbol} != {expected}"
            [s_exp] = s_exp.children
        return SExpression(
            f"lam_{original}",
            (self.from_stitch(s_exp),),
        )


class _UnrepresentableAbstraction(Exception):
    """Raised internally when a stitch abstraction body cannot be represented as a
    NeuroSym ``InitializedSExpression``. ``reason`` is a short category label used
    to diagnose which cases actually occur in practice."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


# Counts, by reason label, of abstractions that ``abstraction_production_from_dreamcoder``
# could not represent (and therefore skipped). Kept as a lightweight diagnostic:
# every skip is a genuine representational limitation (not a bug), so this makes
# it observable which limitations actually bite on a given DSL/run. Truly
# impossible cases (e.g. a free variable) assert instead of landing here.
ABSTRACTION_SKIP_REASONS = collections.Counter()


def _record_abstraction_skip(reason, dreamcoder_abstraction):
    ABSTRACTION_SKIP_REASONS[reason] += 1
    if os.environ.get("NEUROSYM_LOG_ABSTRACTION_SKIPS"):
        print(f"[abstraction-skip:{reason}] {dreamcoder_abstraction}", file=sys.stderr)


def _lambda_symbol(dsl, arity):
    """Symbol of the DSL's ``arity``-argument lambda production (``lam_0`` for a
    single-argument lambda, ``lam_1`` for a two-argument one, ...), or None if the
    DSL has no lambda of that arity.
    """
    for prod in dsl.productions:
        if prod.base_symbol() == "lam" and getattr(prod, "arity", None) == arity:
            return prod.symbol()
    return None


def _abstraction_name_mapping(dsl):
    """Map already-added abstraction productions to atomic placeholder names.

    Each abstraction is added to the DSL under its full DreamCoder ``#(lambda ...)``
    string as its symbol. When a later abstraction is written in terms of an earlier
    one, DreamCoder inlines the earlier one's ``#(lambda ...)`` into the body; this
    mapping lets us rewrite those inlined occurrences to opaque names so they are
    treated as references to the existing production rather than descended into.

    :return: ``(name_mapping, placeholder_to_symbol)`` where ``name_mapping`` is a
        list of ``(placeholder, anonymous)`` pairs and ``placeholder_to_symbol``
        maps each placeholder back to the production symbol.
    """
    anonymous_symbols = [
        prod.symbol() for prod in dsl.productions if prod.symbol().startswith("#")
    ]
    name_mapping = [
        (f"__nested_abstraction_{i}", anonymous)
        for i, anonymous in enumerate(anonymous_symbols)
    ]
    placeholder_to_symbol = {ph: anonymous for ph, anonymous in name_mapping}
    return name_mapping, placeholder_to_symbol


def _dreamcoder_abstraction_to_stitch_body(dreamcoder_abstraction, name_mapping=()):
    """Convert a DreamCoder invented-primitive string into stitch's canonical
    abstraction form, returning ``(body_string, arity)``.

    The leading ``#``-lambdas become abstraction parameters (``#0``, ``#1``,
    ...), with de Bruijn indices remapped accordingly; genuine inner lambdas
    keep their ``$``-de-Bruijn variables. Reuses stitch_core's conversion
    primitives directly -- the ``Abstraction.from_dreamcoder`` wrapper mishandles
    an empty ``name_mapping``.

    ``name_mapping`` is a list of ``(placeholder, anonymous)`` pairs (see
    :py:func:`_abstraction_name_mapping`); each nested ``#(lambda ...)`` invention
    is replaced by its placeholder -- longest first, so a nested invention inside
    another is not mangled -- before the de Bruijn conversion, so the placeholder
    survives as an opaque atom rather than being descended into.
    """
    assert dreamcoder_abstraction.startswith("#"), dreamcoder_abstraction
    body = dreamcoder_abstraction
    for placeholder, anonymous in sorted(name_mapping, key=lambda kv: -len(kv[1])):
        body = body.replace(anonymous, placeholder)
    sexpr = stitch_core.parse(body[1:])
    sexpr, num_lambdas = stitch_core.strip_lambdas(sexpr)
    sexpr = stitch_core.dc_to_stitch_vars(sexpr, 0, num_lambdas)
    return stitch_core.show_sexpr(sexpr), num_lambdas


def _expected_child_arities(dsl, symbol, n_children):
    """For a production applied to ``n_children`` arguments, the arrow arity
    expected at each argument slot -- the number of arguments the function passed
    there takes -- or None for slots that do not expect a function.

    A slot expects a function iff the production's declared type for that argument
    is an :py:class:`~neurosym.types.type.ArrowType`; its arity is the number of
    inputs to that arrow. DreamCoder curries higher-order arguments as nested
    single-argument lambdas, so this is how many of them to fuse back into one
    multi-argument lambda production (see :py:func:`_stitch_body_to_initialized`).
    """
    signature = dsl.get_production(symbol).type_signature()
    arguments = getattr(signature, "arguments", None)
    arities = []
    for i in range(n_children):
        arg_type = (
            arguments[i] if arguments is not None and i < len(arguments) else None
        )
        arities.append(
            len(arg_type.input_type) if isinstance(arg_type, ArrowType) else None
        )
    return arities


def _convert_stitch_node(
    node, expected_arity, dsl, placeholder_to_symbol, binder_levels, new_depth
):
    """Recursively translate a parsed stitch node into a NeuroSym
    ``InitializedSExpression`` (or ``AbstractionIndexParameter``).

    :param expected_arity: The arrow arity the consuming context expects at this
        position (None if it does not expect a function). A curried run of that
        many single-argument lambdas is fused into one multi-argument lambda
        production.
    :param placeholder_to_symbol: Maps placeholder names standing in for inlined
        nested abstractions (see :py:func:`_abstraction_name_mapping`) back to the
        already-added production symbol they reference.
    :param binder_levels: For each lambda binder currently in scope (outermost
        first), the *absolute environment level* it occupies in the translated
        program. DreamCoder curries -- the outermost lambda binds the first
        argument -- whereas a NeuroSym multi-argument lambda numbers ``$k`` by
        argument position, so fusing a group reverses the binders within it; this
        list records the resulting remapping so de Bruijn variables (at any depth)
        can be rewritten correctly.
    :param new_depth: The number of binders in scope in the translated program.
    """
    symbol = node if isinstance(node, str) else node.symbol
    children = () if isinstance(node, str) else node.children

    if symbol in placeholder_to_symbol:
        # A nested invented primitive `#(lambda ...)`, rewritten to a placeholder
        # upstream, that references an already-added abstraction production. Emit a
        # reference to that production, converting any applied arguments.
        referenced = placeholder_to_symbol[symbol]
        arities = _expected_child_arities(dsl, referenced, len(children))
        converted_children = tuple(
            _convert_stitch_node(
                child, arities[i], dsl, placeholder_to_symbol, binder_levels, new_depth
            )
            for i, child in enumerate(children)
        )
        return InitializedSExpression(referenced, converted_children, {})

    if symbol.startswith("#"):
        if not children:
            # `#k` metavariable leaf (abstraction parameter).
            return AbstractionIndexParameter(int(symbol[1:]))
        # A `#`-headed node with children is one of two things that stitch both
        # renders with a bare `#` head: a nested invented primitive `#(lambda ...)`
        # (first child is a lambda) or a metavariable applied to arguments (first
        # child is not). Nested inventions present in the DSL were already rewritten
        # to placeholders upstream, so a `#(lambda ...)` reaching here references an
        # abstraction NOT in this DSL.
        first_child_is_lambda = (
            not isinstance(children[0], str)
            and children[0].symbol in ("lam", "lambda")
        )
        if first_child_is_lambda:
            # Unrepresentable until the referenced abstraction is added (e.g. it was
            # itself skipped, so everything built on it cascades to a skip too).
            raise _UnrepresentableAbstraction("nested_abstraction")
        # A metavariable applied to arguments == a higher-order abstraction, which
        # NeuroSym has no node to represent; callers run Stitch with
        # ``no_curried_metavars`` (see ``single_step_compression`` and lilo's
        # stitch_proposer) to forbid these, so reaching one means a dropped flag.
        raise AssertionError(
            f"metavariable applied to {len(children)} argument(s); callers must run "
            f"Stitch with no_curried_metavars to forbid this"
        )

    if symbol.startswith("$"):
        if children:
            # Applying a bound variable -- not expressible as a NeuroSym variable.
            # (Possible in a DSL whose higher-order primitives pass a function to
            # their callback; never seen in the list domain.)
            raise _UnrepresentableAbstraction("applied_bound_variable")
        old_index = int(symbol[1:])
        # Not a representational limitation but an invariant: every abstraction
        # parameter reference was rewritten to `#k` upstream, so a surviving `$`
        # must be bound by an enclosing inner lambda. A free one means either a
        # malformed abstraction (DreamCoder abstractions are closed) or a bug in
        # the lambda-fusion binder bookkeeping -- fail loudly rather than skip.
        assert 0 <= old_index < len(binder_levels), (
            f"free de Bruijn variable {symbol}: index {old_index} is not bound by "
            f"any of the {len(binder_levels)} enclosing inner lambdas"
        )
        # De Bruijn index counts from the innermost binder; look up that binder's
        # level in the translated program and re-express it as a de Bruijn index.
        new_level = binder_levels[len(binder_levels) - 1 - old_index]
        return InitializedSExpression(f"${new_depth - 1 - new_level}", (), {})

    if symbol in ("lam", "lambda"):
        arity = expected_arity if expected_arity and expected_arity >= 1 else 1
        lam_symbol = _lambda_symbol(dsl, arity)
        if lam_symbol is None:
            raise _UnrepresentableAbstraction("missing_lambda_arity")
        # Peel `arity` nested single-argument lambdas down to the shared body.
        # Callers run Stitch with ``eta_long``, so a higher-order argument arrives
        # as a full run of explicit lambdas; finding fewer than the slot's arrow
        # arity means a dropped flag, not a partially-applied abstraction.
        body = node
        for _ in range(arity):
            assert not isinstance(body, str) and body.symbol in ("lam", "lambda"), (
                f"expected {arity} nested lambdas for an arity-{arity} argument "
                f"but found fewer; callers must run Stitch with eta_long"
            )
            (body,) = body.children
        # The fused lambda binds `arity` new levels. In a native multi-argument
        # lambda, argument position k is `$k`; the outermost curried lambda binds
        # the first argument, so map outermost -> position 0 by reversing the group.
        fused_levels = [new_depth + (arity - 1 - p) for p in range(arity)]
        converted_body = _convert_stitch_node(
            body, None, dsl, placeholder_to_symbol, binder_levels + fused_levels,
            new_depth + arity,
        )
        return InitializedSExpression(lam_symbol, (converted_body,), {})

    if not children:
        # A nullary primitive leaf (e.g. `0`, `empty`, `true`).
        return InitializedSExpression(symbol, (), {})

    arities = _expected_child_arities(dsl, symbol, len(children))
    converted_children = tuple(
        _convert_stitch_node(
            child, arities[i], dsl, placeholder_to_symbol, binder_levels, new_depth
        )
        for i, child in enumerate(children)
    )
    return InitializedSExpression(symbol, converted_children, {})


def _stitch_body_to_initialized(node, dsl, placeholder_to_symbol=()):
    """Translate a parsed stitch abstraction body into a NeuroSym
    ``InitializedSExpression``.

    ``#N`` -> ``AbstractionIndexParameter(N)`` and bare-string leaves ->
    zero-child nodes. DreamCoder curries every lambda, but the NeuroSym list DSL
    types higher-order primitives (``mapi``, ``reducei``, ...) with uncurried
    multi-argument arrows, so a run of nested single-argument lambdas feeding such
    a primitive is fused into the DSL's corresponding multi-argument lambda
    production, remapping the enclosed de Bruijn variables. ``placeholder_to_symbol``
    (see :py:func:`_abstraction_name_mapping`) maps placeholder tokens standing in
    for inlined nested abstractions back to the productions they reference.

    Raises :py:class:`_UnrepresentableAbstraction` (with a ``reason`` label) if the
    body cannot be represented -- an applied bound variable, an unknown primitive, a
    required multi-argument lambda production missing from the DSL, or a nested
    abstraction not yet added to the DSL. Cases that cannot occur given how callers
    invoke Stitch instead trip an assertion: a free variable (abstractions are
    closed), a higher-order metavariable application (forbidden by
    ``no_curried_metavars``), and a partially-applied higher-order argument (ruled
    out by ``eta_long``).
    """
    try:
        return _convert_stitch_node(node, None, dsl, dict(placeholder_to_symbol), [], 0)
    except KeyError as e:
        # dsl.get_production(...) failed: a primitive in the abstraction is absent
        # from this DSL.
        raise _UnrepresentableAbstraction("unknown_primitive") from e


def abstraction_production_from_dreamcoder(dsl, name, dreamcoder_abstraction):
    """Build a NeuroSym ``AbstractionProduction`` from a DreamCoder/Stitch
    invented-primitive string, e.g.
    ``#(lambda (all_1 (lambda (is-prime $0)) $0))``.

    The abstraction's own parameters (the leading ``#``-lambdas) become
    ``AbstractionIndexParameter``s; genuine inner lambdas (arguments to
    higher-order primitives like ``map``/``filter``/``all``) become the DSL's
    lambda production of matching arity -- a curried run of single-argument
    lambdas feeding a multi-argument primitive (e.g. ``mapi``/``reducei``) is
    fused into one multi-argument lambda. A nested invented primitive (one
    abstraction inlined into another) is rewritten to a reference to the
    already-added production it names. Argument and return types are inferred
    from the body via :py:meth:`DSL.compute_type_abs`.

    :param dsl: The base DSL the abstraction is written over.
    :param name: The symbol to give the resulting production (typically the
        DreamCoder abstraction string, so it matches the grammar it came from).
    :param dreamcoder_abstraction: The DreamCoder invented-primitive string.

    :return: The ``AbstractionProduction``, or None if the abstraction cannot be
        represented or typed in ``dsl`` -- an unknown primitive, failed type
        unification, a higher-order parameter application, or free variables.
    """
    name_mapping, placeholder_to_symbol = _abstraction_name_mapping(dsl)
    try:
        body_string, _arity = _dreamcoder_abstraction_to_stitch_body(
            dreamcoder_abstraction, name_mapping
        )
        parsed = parse_s_expression(body_string, for_stitch=True)
    except (stitch_core.InvalidAbstractionException, ValueError):
        # Unparseable / free variables / a higher-order metavariable applied to
        # arguments (its body does not render as a single s-expression). None of
        # these are representable NeuroSym abstractions.
        _record_abstraction_skip("unparseable", dreamcoder_abstraction)
        return None
    try:
        body = _stitch_body_to_initialized(parsed, dsl, placeholder_to_symbol)
    except _UnrepresentableAbstraction as e:
        _record_abstraction_skip(e.reason, dreamcoder_abstraction)
        return None
    try:
        type_with_env = dsl.compute_type_abs(body)
    except KeyError:
        # A primitive referenced by the abstraction is absent from this DSL.
        _record_abstraction_skip("unknown_primitive_in_typing", dreamcoder_abstraction)
        return None
    if type_with_env is None:
        _record_abstraction_skip("type_unification_failed", dreamcoder_abstraction)
        return None

    if isinstance(type_with_env.typ, ArrowType):
        type_signature = FunctionTypeSignature.from_type(type_with_env.typ)
    else:
        # Build a function type from the inferred environment. Each abstraction
        # parameter is keyed by its stitch metavariable index (`#0`, `#1`, ...),
        # and DreamCoder applies arguments in that order (`#0` is the first
        # applied argument), so order arguments by ascending index.
        env_elements = type_with_env.env._elements
        argument_types = []
        for env_index in sorted(env_elements.keys()):
            env_typ = env_elements[env_index]
            if isinstance(env_typ, tuple):
                assert len(env_typ) == 1, env_typ
                env_typ = env_typ[0]
            argument_types.append(env_typ)
        type_signature = FunctionTypeSignature(argument_types, type_with_env.typ)

    return AbstractionProduction(name, type_signature, body)


def single_step_compression(dsl: DSL, programs: List[SExpression]):
    """
    Run single step of compression on a list of programs.

    :param dsl: The DSL the programs are written in.
    :param programs: The programs to compress.

    :return: The DSL with up to 1 additional abstraction, and the programs
        rewritten to use the new abstractions.
    """
    programs_orig = programs
    rewriter = _StitchLambdaRewriter(dsl)
    programs = [rewriter.to_stitch(prog) for prog in programs]
    rendered = [render_s_expression(prog, for_stitch=True) for prog in programs]
    res = stitch_core.compress(
        rendered,
        1,
        no_curried_bodies=True,
        no_curried_metavars=True,
        fused_lambda_tags=rewriter.fused_lambda_tags,
        abstraction_prefix=_next_symbol(dsl),
    )
    if not res.abstractions:
        return dsl, programs_orig
    abstr = res.abstractions[-1]
    rewritten = [
        rewriter.from_stitch(
            parse_s_expression(x, should_not_be_leaf={abstr.name}, for_stitch=True)
        )
        for x in res.rewritten
    ]
    user = next(x for x in rewritten if abstr.name in symbols_for_program(x))
    abstr_body = rewriter.from_stitch(
        parse_s_expression(abstr.body, should_not_be_leaf={abstr.name}, for_stitch=True)
    )
    prod = _compute_abstraction_production(dsl, user, abstr.name, abstr_body)
    dsl2 = dsl.add_productions(prod)
    return dsl2, rewritten


def multi_step_compression(dsl: DSL, programs: List[SExpression], iterations: int):
    """
    Run multiple steps of compression on a list of programs.

    :param dsl: The DSL the programs are written in.
    :param programs: The programs to compress.
    :param iterations: The number of iterations to run.

    :return: The DSL with up to `iterations` new abstraction productions, and the programs
        rewritten to use the new abstractions.
    """
    for _ in range(iterations):
        dsl, programs = single_step_compression(dsl, programs)
    return dsl, programs
