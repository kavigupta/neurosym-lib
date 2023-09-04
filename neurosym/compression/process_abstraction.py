import itertools
from typing import Union
import stitch_core
from neurosym.dsl.dsl import DSL
from neurosym.dsl.abstraction import AbstractionIndexParameter, AbstractionProduction
from neurosym.programs.s_expression import SExpression
from neurosym.programs.s_expression_render import (
    parse_s_expression,
    render_s_expression,
    symbols,
)
from neurosym.types.type_signature import ConcreteTypeSignature


def compute_abstraction_production(
    dsl,
    s_expression_using: SExpression,
    abstr: stitch_core.Abstraction,
):
    """
    Compute the type of an abstraction production.

    Args:
        dsl: The DSL.
        s_expression_using: An SExpression using the abstraction.
        abstr: The original abstraction.

    Returns an AbstractionProduction corresponding to the abstraction.
    """
    body_se = parse_s_expression(abstr.body, {abstr.name})
    body_se = inject_parameters(body_se)
    usage = next(x for x in s_expression_using.postorder if x.symbol == abstr.name)
    type_arguments = [dsl.compute_type(x) for x in usage.children]
    type_out = dsl.compute_type(
        body_se,
        lambda x: type_arguments[x.index]
        if isinstance(x, AbstractionIndexParameter)
        else None,
    ).typ
    type_signature = ConcreteTypeSignature(type_arguments, type_out)

    return AbstractionProduction(abstr.name, type_signature, body_se)


def inject_parameters(s_expression: Union[SExpression, str]):
    """
    Inject parameters into an SExpression, replacing leaves of the form #N with
    AbstractionIndexParameter(N).
    """
    if isinstance(s_expression, str):
        assert s_expression.startswith("#")
        return AbstractionIndexParameter(int(s_expression[1:]))
    return SExpression(
        s_expression.symbol,
        tuple(inject_parameters(x) for x in s_expression.children),
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


def single_step_compression(dsl, programs):
    """
    Run a single step of compression on a list of programs.
    """
    rendered = [render_s_expression(prog, for_stitch=True) for prog in programs]
    res = stitch_core.compress(
        rendered,
        1,
        no_curried_bodies=True,
        no_curried_metavars=True,
        abstraction_prefix=next_symbol(dsl),
    )
    abstr = res.abstractions[-1]
    rewritten = (parse_s_expression(x, {abstr.name}) for x in res.rewritten)
    user = next(x for x in rewritten if abstr.name in symbols(x))
    prod = compute_abstraction_production(dsl, user, abstr)
    rewritten = stitch_core.rewrite(rendered, [abstr]).rewritten
    rewritten = [parse_s_expression(x, {abstr.name}) for x in rewritten]
    dsl2 = DSL(dsl.productions + [prod])
    return dsl2, rewritten


def multi_step_compression(dsl, programs, iterations):
    """
    Run multiple steps of compression on a list of programs.
    """
    for _ in range(iterations):
        dsl, programs = single_step_compression(dsl, programs)
    return dsl, programs
