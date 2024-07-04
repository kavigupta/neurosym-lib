import itertools
from typing import List, Union

import stitch_core

from neurosym.dsl.dsl import DSL

from ..dsl.abstraction import AbstractionIndexParameter, AbstractionProduction
from ..programs.s_expression import SExpression
from ..programs.s_expression_render import (
    parse_s_expression,
    render_s_expression,
    symbols_for_program,
)
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
    usage = next(x for x in s_expression_using.postorder if x.symbol == abstr_name)
    type_arguments = [dsl.compute_type(x) for x in usage.children]
    type_out = dsl.compute_type(
        abstr_body,
        lambda x: (
            type_arguments[x.index]
            if isinstance(x, AbstractionIndexParameter)
            else None
        ),
    ).typ
    type_signature = FunctionTypeSignature([x.typ for x in type_arguments], type_out)

    return AbstractionProduction(abstr_name, type_signature, abstr_body)


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
        self.zero_arg_lambda_symbol = _next_symbol(dsl)

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
    dsl2 = dsl.add_production(prod)
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
