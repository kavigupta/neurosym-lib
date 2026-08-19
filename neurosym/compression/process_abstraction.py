import itertools
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
    NeuroSym ``InitializedSExpression`` -- e.g. a higher-order metavariable applied
    to arguments, an applied/free variable, or a curried lambda whose required
    multi-argument lambda production is absent from the DSL."""


def _lambda_symbol(dsl, arity):
    """Symbol of the DSL's ``arity``-argument lambda production (``lam_0`` for a
    single-argument lambda, ``lam_1`` for a two-argument one, ...), or None if the
    DSL has no lambda of that arity.
    """
    for prod in dsl.productions:
        if prod.base_symbol() == "lam" and getattr(prod, "arity", None) == arity:
            return prod.symbol()
    return None


def _dreamcoder_abstraction_to_stitch_body(dreamcoder_abstraction):
    """Convert a DreamCoder invented-primitive string into stitch's canonical
    abstraction form, returning ``(body_string, arity)``.

    The leading ``#``-lambdas become abstraction parameters (``#0``, ``#1``,
    ...), with de Bruijn indices remapped accordingly; genuine inner lambdas
    keep their ``$``-de-Bruijn variables. Reuses stitch_core's conversion
    primitives directly -- the ``Abstraction.from_dreamcoder`` wrapper mishandles
    an empty ``name_mapping``.
    """
    assert dreamcoder_abstraction.startswith("#"), dreamcoder_abstraction
    sexpr = stitch_core.parse(dreamcoder_abstraction[1:])
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


def _convert_stitch_node(node, expected_arity, dsl, binder_levels, new_depth):
    """Recursively translate a parsed stitch node into a NeuroSym
    ``InitializedSExpression`` (or ``AbstractionIndexParameter``).

    :param expected_arity: The arrow arity the consuming context expects at this
        position (None if it does not expect a function). A curried run of that
        many single-argument lambdas is fused into one multi-argument lambda
        production.
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

    if symbol.startswith("#"):
        if children:
            # A metavariable applied to arguments (higher-order abstraction) or a
            # nested invented primitive `#(lambda ...)` -- neither is expressible.
            raise _UnrepresentableAbstraction()
        return AbstractionIndexParameter(int(symbol[1:]))

    if symbol.startswith("$"):
        if children:
            # Applying a bound variable -- not expressible as a NeuroSym variable.
            raise _UnrepresentableAbstraction()
        old_index = int(symbol[1:])
        if not 0 <= old_index < len(binder_levels):
            # A free variable (not bound by any enclosing inner lambda).
            raise _UnrepresentableAbstraction()
        # De Bruijn index counts from the innermost binder; look up that binder's
        # level in the translated program and re-express it as a de Bruijn index.
        new_level = binder_levels[len(binder_levels) - 1 - old_index]
        return InitializedSExpression(f"${new_depth - 1 - new_level}", (), {})

    if symbol in ("lam", "lambda"):
        arity = expected_arity if expected_arity and expected_arity >= 1 else 1
        lam_symbol = _lambda_symbol(dsl, arity)
        if lam_symbol is None:
            raise _UnrepresentableAbstraction()
        # Peel `arity` nested single-argument lambdas down to the shared body.
        body = node
        for _ in range(arity):
            if isinstance(body, str) or body.symbol not in ("lam", "lambda"):
                # Fewer lambdas than the expected arrow arity (partial application).
                raise _UnrepresentableAbstraction()
            (body,) = body.children
        # The fused lambda binds `arity` new levels. In a native multi-argument
        # lambda, argument position k is `$k`; the outermost curried lambda binds
        # the first argument, so map outermost -> position 0 by reversing the group.
        fused_levels = [new_depth + (arity - 1 - p) for p in range(arity)]
        converted_body = _convert_stitch_node(
            body, None, dsl, binder_levels + fused_levels, new_depth + arity
        )
        return InitializedSExpression(lam_symbol, (converted_body,), {})

    if not children:
        # A nullary primitive leaf (e.g. `0`, `empty`, `true`).
        return InitializedSExpression(symbol, (), {})

    arities = _expected_child_arities(dsl, symbol, len(children))
    converted_children = tuple(
        _convert_stitch_node(child, arities[i], dsl, binder_levels, new_depth)
        for i, child in enumerate(children)
    )
    return InitializedSExpression(symbol, converted_children, {})


def _stitch_body_to_initialized(node, dsl):
    """Translate a parsed stitch abstraction body into a NeuroSym
    ``InitializedSExpression``.

    ``#N`` -> ``AbstractionIndexParameter(N)`` and bare-string leaves ->
    zero-child nodes. DreamCoder curries every lambda, but the NeuroSym list DSL
    types higher-order primitives (``mapi``, ``reducei``, ...) with uncurried
    multi-argument arrows, so a run of nested single-argument lambdas feeding such
    a primitive is fused into the DSL's corresponding multi-argument lambda
    production, remapping the enclosed de Bruijn variables.

    Returns None if the body cannot be represented (a higher-order metavariable
    applied to arguments, an applied/free variable, an unknown primitive, or a
    required multi-argument lambda production missing from the DSL).
    """
    try:
        return _convert_stitch_node(node, None, dsl, [], 0)
    except (KeyError, _UnrepresentableAbstraction):
        return None


def abstraction_production_from_dreamcoder(dsl, name, dreamcoder_abstraction):
    """Build a NeuroSym ``AbstractionProduction`` from a DreamCoder/Stitch
    invented-primitive string, e.g.
    ``#(lambda (all_1 (lambda (is-prime $0)) $0))``.

    The abstraction's own parameters (the leading ``#``-lambdas) become
    ``AbstractionIndexParameter``s; genuine inner lambdas (arguments to
    higher-order primitives like ``map``/``filter``/``all``) become the DSL's
    lambda production of matching arity -- a curried run of single-argument
    lambdas feeding a multi-argument primitive (e.g. ``mapi``/``reducei``) is
    fused into one multi-argument lambda. Argument and return types are inferred
    from the body via :py:meth:`DSL.compute_type_abs`.

    :param dsl: The base DSL the abstraction is written over.
    :param name: The symbol to give the resulting production (typically the
        DreamCoder abstraction string, so it matches the grammar it came from).
    :param dreamcoder_abstraction: The DreamCoder invented-primitive string.

    :return: The ``AbstractionProduction``, or None if the abstraction cannot be
        represented or typed in ``dsl`` -- an unknown primitive, failed type
        unification, a higher-order parameter application, or free variables.
    """
    try:
        body_string, _arity = _dreamcoder_abstraction_to_stitch_body(
            dreamcoder_abstraction
        )
        parsed = parse_s_expression(body_string, for_stitch=True)
    except (stitch_core.InvalidAbstractionException, ValueError):
        # Unparseable / free variables / a higher-order metavariable applied to
        # arguments (its body does not render as a single s-expression). None of
        # these are representable NeuroSym abstractions.
        return None
    body = _stitch_body_to_initialized(parsed, dsl)
    if body is None:
        return None
    try:
        type_with_env = dsl.compute_type_abs(body)
    except KeyError:
        # A primitive referenced by the abstraction is absent from this DSL.
        return None
    if type_with_env is None:
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
