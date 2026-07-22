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


def _single_arg_lambda_symbol(dsl):
    """Symbol of the DSL's single-argument lambda production (e.g. ``lam_0``).

    DreamCoder lambdas are always single-argument (curried), so every inner
    lambda in a DreamCoder abstraction maps to this production. Returns None if
    the DSL has no single-argument lambda.
    """
    for prod in dsl.productions:
        if prod.base_symbol() == "lam" and getattr(prod, "arity", None) == 1:
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


def _stitch_body_to_initialized(node, lam_symbol):
    """Translate a parsed stitch abstraction body into a NeuroSym
    ``InitializedSExpression``.

    ``#N`` -> ``AbstractionIndexParameter(N)``, single-argument lambdas ->
    ``lam_symbol``, and bare-string leaves -> zero-child nodes.

    Returns None if a parameter is applied to arguments (a higher-order
    metavariable), which NeuroSym's abstraction representation cannot express.
    """
    if isinstance(node, str):
        if node.startswith("#"):
            return AbstractionIndexParameter(int(node[1:]))
        symbol = lam_symbol if node in ("lam", "lambda") else node
        return InitializedSExpression(symbol, (), {})
    if node.symbol.startswith("#"):
        # `#N` applied to arguments == higher-order parameter application.
        return None
    symbol = lam_symbol if node.symbol in ("lam", "lambda") else node.symbol
    children = []
    for child in node.children:
        converted = _stitch_body_to_initialized(child, lam_symbol)
        if converted is None:
            return None
        children.append(converted)
    return InitializedSExpression(symbol, tuple(children), {})


def abstraction_production_from_dreamcoder(dsl, name, dreamcoder_abstraction):
    """Build a NeuroSym ``AbstractionProduction`` from a DreamCoder/Stitch
    invented-primitive string, e.g.
    ``#(lambda (all_1 (lambda (is-prime $0)) $0))``.

    The abstraction's own parameters (the leading ``#``-lambdas) become
    ``AbstractionIndexParameter``s; genuine inner lambdas (arguments to
    higher-order primitives like ``map``/``filter``/``all``) become the DSL's
    single-argument lambda production. Argument and return types are inferred
    from the body via :py:meth:`DSL.compute_type_abs`.

    :param dsl: The base DSL the abstraction is written over.
    :param name: The symbol to give the resulting production (typically the
        DreamCoder abstraction string, so it matches the grammar it came from).
    :param dreamcoder_abstraction: The DreamCoder invented-primitive string.

    :return: The ``AbstractionProduction``, or None if the abstraction cannot be
        represented or typed in ``dsl`` -- an unknown primitive, failed type
        unification, a higher-order parameter application, or free variables.
    """
    lam_symbol = _single_arg_lambda_symbol(dsl)
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
    body = _stitch_body_to_initialized(parsed, lam_symbol)
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
        # Build a function type from the inferred environment. Environment index
        # 0 is the most recent binder, so order arguments by descending index
        # (outermost parameter first).
        env_elements = type_with_env.env._elements
        argument_types = []
        for env_index in sorted(env_elements.keys(), reverse=True):
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
