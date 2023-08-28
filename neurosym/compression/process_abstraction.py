import itertools
import stitch_core
from neurosym.dsl.dsl import DSL
from neurosym.dsl.production import AbstractionProduction
from neurosym.programs.s_expression import SExpression
from neurosym.programs.s_expression_render import (
    parse_s_expression,
    render_s_expression,
    symbols,
)
from neurosym.types.type import Type
from neurosym.types.type_signature import ConcreteTypeSignature


def compute_abstraction_type(
    dsl,
    target_type: Type,
    s_expression_using: SExpression,
    abstr_original: stitch_core.Abstraction,
):
    """
    Compute the type of an abstraction production.

    Args:
        dsl: The DSL.
        target_type: The target type.
        s_expression_using: The SExpression to use.
        abstr_original: The original abstraction.

    Returns:
        A stitch.Abstraction representing the updated abstraction as well
        as an AbstractionProduction we can use.
    """
    names = {abstr_original.name}
    body_se = parse_s_expression(abstr_original.body, names)
    proper_arity = dsl.arity(body_se.symbol)
    for addtl_var in range(len(body_se.children), proper_arity):
        body_se.children.append(f"#{addtl_var}")
    abstr = stitch_core.Abstraction(
        abstr_original.name, render_s_expression(body_se, for_stitch=True), proper_arity
    )
    usage = next(
        x for x in s_expression_using.postorder if x.symbol == abstr_original.name
    )
    type_arguments = [dsl.compute_type(x) for x in usage.children]
    by_type = {f"#{i}": typ for i, typ in enumerate(type_arguments)}
    type_out = dsl.compute_type(
        body_se, lambda x: by_type.get(x, None) if isinstance(x, str) else None
    )

    return abstr, AbstractionProduction(
        abstr.name,
        ConcreteTypeSignature(type_arguments, type_out),
        body_se,
    )


def next_symbol(dsl):
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


def single_step_compression(dsl, programs, out_t):
    rendered = [render_s_expression(prog, for_stitch=True) for prog in programs]
    res = stitch_core.compress(
        rendered,
        1,
        no_curried_bodies=True,
        no_curried_metavars=True,
        abstraction_prefix=next_symbol(dsl),
    )
    abstr_original = res.abstractions[-1]
    rewritten = (parse_s_expression(x, {abstr_original.name}) for x in res.rewritten)
    user = next(x for x in rewritten if abstr_original.name in symbols(x))
    abstr, prod = compute_abstraction_type(dsl, out_t, user, abstr_original)
    rewritten = stitch_core.rewrite(rendered, [abstr]).rewritten
    rewritten = [parse_s_expression(x, {abstr_original.name}) for x in rewritten]
    dsl2 = DSL(dsl.productions + [prod])
    return dsl2, rewritten


def multi_step_compression(dsl, programs, out_t, iterations):
    for _ in range(iterations):
        dsl, programs = single_step_compression(dsl, programs, out_t)
    return dsl, programs
