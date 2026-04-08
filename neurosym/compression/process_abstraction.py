import itertools
import uuid
from typing import List, Union

import stitch_core

from neurosym.dsl.dsl import DSL

from ..dsl.abstraction import AbstractionIndexParameter, AbstractionProduction
from ..programs.s_expression import SExpression, postorder
from ..programs.s_expression_render import (
    parse_s_expression,
    render_s_expression,
    symbols_for_program,
)
from ..types.type import UnificationError
from ..types.type_signature import FunctionTypeSignature


def _compute_abstraction_production(
    dsl,
    s_expression_using: SExpression,
    abstr_name: str,
    abstr_body: SExpression,
):
    """
    Compute the type of an abstraction production.

    Uses top-down type inference (unify_return) from the program root to
    the abstraction call site to determine the return type and environment.
    Argument types are computed from the usage children using the resolved
    environment, avoiding any polymorphic type variables.

    Args:
        dsl: The DSL.
        s_expression_using: An SExpression using the abstraction.
        abstr_name: The name of the abstraction being created.
        abstr_body: The body of the abstraction (with #N placeholders).

    Returns an AbstractionProduction corresponding to the abstraction.
    """
    abstr_body = _inject_parameters(abstr_body)

    # Top-down: walk from the root to the call site to get the concrete
    # expected type and environment at that position.
    call_site_twe = _find_type_at_call_site(
        dsl, s_expression_using, abstr_name, abstr_body
    )
    usage = next(x for x in postorder(s_expression_using) if x.symbol == abstr_name)

    # Compute each argument's type using the environment from the call site
    # so that polymorphic variables ($N) resolve to concrete types.
    arg_types = [
        _compute_type_in_env(dsl, child, call_site_twe.env).typ
        for child in usage.children
    ]
    type_signature = FunctionTypeSignature(arg_types, call_site_twe.typ)

    return AbstractionProduction(abstr_name, type_signature, abstr_body)


def _find_type_at_call_site(dsl, program, target_symbol, abstr_body):
    """Walk the program tree top-down to find the expected type at target_symbol.

    Uses unify_return at each production to propagate the expected type
    from the root down to the target node. Validates the result by checking
    the abstraction body type-checks with the inferred type.
    """
    from ..types.type_with_environment import (  # pylint: disable=import-outside-toplevel
        StrictEnvironment,
        TypeWithEnvironment,
    )

    usage = next(x for x in postorder(program) if x.symbol == target_symbol)
    root_twes = [
        TypeWithEnvironment(t, StrictEnvironment.empty()) for t in dsl.valid_root_types
    ]
    for root_twe in root_twes:
        result = _find_type_topdown(dsl, program, target_symbol, root_twe)
        if result is None:
            continue
        # Validate: check the body type-checks with this return type and arg types
        arg_types = [
            _compute_type_in_env(dsl, child, result.env).typ for child in usage.children
        ]
        try:
            body_twe = dsl.compute_type(
                abstr_body,
                lambda x, _at=arg_types, _env=result.env: (
                    TypeWithEnvironment(_at[x.index], _env)
                    if isinstance(x, AbstractionIndexParameter)
                    else None
                ),
            )
            # Body type may have type variables for unused lambda args;
            # check compatibility via unification rather than exact equality.
            body_twe.typ.unify(result.typ)
            return result
        except (ValueError, AssertionError, AttributeError, KeyError, UnificationError):
            continue
    raise ValueError(
        f"Could not find {target_symbol} reachable from any root type in {dsl.valid_root_types}"
    )


def _find_type_topdown(dsl, node, target_symbol, expected_twe):
    """Recursive top-down walk. Returns the TWE at target_symbol, or None."""
    if node.symbol == target_symbol:
        return expected_twe

    try:
        prod = dsl.get_production(node.symbol)
    except KeyError:
        return None

    child_twes = prod.type_signature().unify_return(expected_twe)
    if child_twes is None or len(child_twes) != len(node.children):
        return None

    for child, child_twe in zip(node.children, child_twes):
        result = _find_type_topdown(dsl, child, target_symbol, child_twe)
        if result is not None:
            return result
    return None


def _compute_type_in_env(dsl, program, env):
    """Compute the type of a program, resolving variables using the given environment."""
    from ..types.type_signature import (
        VariableTypeSignature,
    )  # pylint: disable=import-outside-toplevel
    from ..types.type_with_environment import (
        TypeWithEnvironment,
    )  # pylint: disable=import-outside-toplevel

    def lookup(node):
        try:
            prod = dsl.get_production(node.symbol)
        except KeyError:
            return None
        sig = prod.type_signature()
        if isinstance(sig, VariableTypeSignature):
            idx = sig.index_in_env
            typ = env[idx] if idx < len(env) else None
            if typ is not None:
                return TypeWithEnvironment(typ, env)
        return None

    return dsl.compute_type(program, lookup)


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
