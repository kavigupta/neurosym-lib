import functools
import itertools
import types
import unittest
from fractions import Fraction

import numpy as np

import neurosym as ns
from neurosym.compression.process_abstraction import (
    _convert_stitch_node,
    abstraction_production_from_dreamcoder,
)
from neurosym.dsl.abstraction import AbstractionProduction, _with_index_parameters
from neurosym.examples.dreamcoder.list_example import list_dslf
from neurosym.program_dist.bigram import BigramProgramDistributionFamily
from neurosym.types.type import AtomicType
from neurosym.types.type_signature import FunctionTypeSignature
from neurosym.types.type_string_repr import parse_type, render_type
from neurosym.types.type_with_environment import Environment, TypeWithEnvironment

_LIST_OUTPUT_TYPES = [
    "[i] -> [i]", "[i] -> b", "[i] -> i", "b -> [i]", "b -> b",
    "b -> i", "i -> [i]", "i -> b", "i -> i",
]


@functools.lru_cache(maxsize=None)
def _list_dsl():
    # Building the list DSL is expensive (~20s); it is immutable and
    # abstraction_production_from_dreamcoder does not mutate it, so cache it.
    return list_dslf(*sorted(_LIST_OUTPUT_TYPES)).finalize()


def _render_signature(signature):
    return render_type(signature.astype())


class TestAbstractionParameterTyping(unittest.TestCase):
    """Typing DreamCoder abstractions whose parameters are used at various
    depths and under various parents. Regression coverage for a crash where an
    abstraction parameter (a stitch metavariable ``#k``) appeared as the body of
    an inner lambda -- ``DSL.compute_type_abs`` assumed the parent was always a
    function-typed production and asserted on the lambda's ``LambdaTypeSignature``.
    """

    def _signature_for(self, abstraction):
        prod = abstraction_production_from_dreamcoder(
            _list_dsl(), abstraction, abstraction
        )
        return None if prod is None else prod.type_signature()

    def test_single_parameter(self):
        sig = self._signature_for("#(lambda (gt? 1 (sum $0)))")
        self.assertEqual(_render_signature(sig), "[i] -> b")

    def test_two_parameters_ordered_by_application(self):
        # $1 is the first-applied argument (#0), $0 the second (#1); a previous
        # bug both crashed here and would have reversed the argument order.
        sig = self._signature_for("#(lambda (lambda (gt? $1 (sum $0))))")
        self.assertEqual(_render_signature(sig), "(i, [i]) -> b")

    def test_parameter_as_lambda_body(self):
        # Parameter $3 is the body of a nested inner lambda passed to mapi_1. This
        # used to raise AssertionError in compute_type_abs. The nested single-arg
        # lambdas are fused into mapi_1's 2-argument callback, so it now types:
        # the callback ignores its args and returns parameter #1, mapping (range
        # #0) to a list of #1 -- i.e. (int, t) -> [t].
        sig = self._signature_for(
            "#(lambda (lambda (lambda (mapi_1 (lambda (lambda $3)) (range $2)))))"
        )
        self.assertIsNotNone(sig)
        self.assertEqual(len(sig.arguments), 2)
        self.assertEqual(render_type(sig.arguments[0]), "i")
        # return type is a list of the (polymorphic) second parameter.
        self.assertEqual(sig.return_type, ns.ListType(sig.arguments[1]))


class TestCurriedLambdaFusion(unittest.TestCase):
    """DreamCoder curries higher-order arguments as nested single-argument
    lambdas, but the list DSL types ``mapi``/``reducei`` with uncurried
    multi-argument arrows. The conversion fuses such lambda runs into the DSL's
    matching multi-argument lambda production; these tests pin the resulting types
    and the correctness of the de Bruijn remapping.
    """

    def _signature_for(self, abstraction):
        prod = abstraction_production_from_dreamcoder(
            _list_dsl(), abstraction, abstraction
        )
        return None if prod is None else prod.type_signature()

    def test_constant_callback(self):
        # mapi_1 with a callback that ignores both args and returns 0.
        sig = self._signature_for("#(lambda (mapi_1 (lambda (lambda 0)) $0))")
        self.assertEqual(_render_signature(sig), "[i] -> [i]")

    def test_callback_uses_both_fused_arguments(self):
        # (+ $1 $0) uses mapi_1's index ($1) and element ($0).
        sig = self._signature_for("#(lambda (mapi_1 (lambda (lambda (+ $1 $0))) $0))")
        self.assertEqual(_render_signature(sig), "[i] -> [i]")

    def test_three_argument_callback_reducei(self):
        # reducei_1 takes a 3-argument callback (index, acc, element) -> fused into
        # the DSL's 3-argument lambda (lam_2).
        sig = self._signature_for(
            "#(lambda (reducei_1 (lambda (lambda (lambda (+ $1 $0)))) 0 $0))"
        )
        self.assertEqual(_render_signature(sig), "[i] -> i")

    def test_fused_argument_order_is_not_symmetric(self):
        # mapi_3's callback binds (index : int, element : [i]). index-into-list
        # requires (int, list); it types only if the fused binders keep that order.
        ok = self._signature_for(
            "#(lambda (mapi_3 (lambda (lambda (index $1 $0))) $0))"
        )
        self.assertIsNotNone(ok)
        self.assertEqual(len(ok.arguments), 1)
        self.assertEqual(render_type(ok.arguments[0]), "[[i]]")
        # Swapping the two references is ill-typed (index wants int first).
        swapped = self._signature_for(
            "#(lambda (mapi_3 (lambda (lambda (index $0 $1))) $0))"
        )
        self.assertIsNone(swapped)

    def test_deeply_nested_variable_reference(self):
        # Inner mapi_1 callback references the OUTER mapi_3 index via $3, crossing
        # the fused inner group; exercises the depth-aware de Bruijn remap.
        sig = self._signature_for(
            "#(lambda (mapi_3 (lambda (lambda "
            "(mapi_1 (lambda (lambda (+ $3 $0))) $0))) $0))"
        )
        self.assertEqual(_render_signature(sig), "[[i]] -> [[i]]")

    def test_free_variable_asserts_rather_than_skipping(self):
        # DreamCoder abstractions are closed and every parameter reference is
        # rewritten to `#k` upstream, so a surviving free `$` is impossible for
        # valid input -- it signals malformed input or a binder-tracking bug and
        # must fail loudly, not silently skip. `$5` here is bound by nothing.
        node = types.SimpleNamespace(
            symbol="sum",
            children=(types.SimpleNamespace(symbol="$5", children=()),),
        )
        with self.assertRaises(AssertionError):
            _convert_stitch_node(node, None, _list_dsl(), {}, [], 0)

    def test_applied_metavariable_asserts(self):
        # A metavariable applied to arguments (higher-order abstraction) is
        # forbidden by Stitch's no_curried_metavars flag, so reaching one signals
        # a dropped flag -- assert rather than skip. Stitch renders an applied
        # metavariable with a bare `#` head and a non-lambda first child (e.g.
        # `#(1)` for `($0 1)`), distinct from a nested `#(lambda ...)` invention.
        node = types.SimpleNamespace(
            symbol="#",
            children=(types.SimpleNamespace(symbol="1", children=()),),
        )
        with self.assertRaises(AssertionError):
            _convert_stitch_node(node, None, _list_dsl(), {}, [], 0)

    def test_nested_abstraction_references_added_production(self):
        # One abstraction defined in terms of another arrives with the inner one
        # inlined as `#(lambda ...)`. Once that inner abstraction is a production in
        # the DSL, the outer one should reference it (not skip, not crash).
        inner = "#(lambda (gt? 1 (sum $0)))"
        outer = "#(lambda (#(lambda (gt? 1 (sum $0))) $0))"  # outer applies inner

        dsl = _list_dsl()
        # Before the inner abstraction is added, the outer one cannot be represented.
        self.assertIsNone(
            abstraction_production_from_dreamcoder(dsl, outer, outer)
        )

        inner_prod = abstraction_production_from_dreamcoder(dsl, inner, inner)
        dsl = dsl.add_productions(inner_prod)
        outer_prod = abstraction_production_from_dreamcoder(dsl, outer, outer)
        # Outer applies inner (an `[i] -> b`) to its own `[i]` argument, so it has
        # the same type, and its body references the inner production's symbol.
        self.assertEqual(_render_signature(outer_prod.type_signature()), "[i] -> b")
        self.assertIn(inner, ns.render_s_expression(outer_prod._body))

    def test_nested_abstraction_inside_fused_lambda(self):
        # The referenced abstraction can appear inside a fused multi-arg lambda,
        # applied to a remapped bound variable.
        inner = "#(lambda (gt? 1 (sum $0)))"
        outer = "#(lambda (mapi_3 (lambda (lambda (#(lambda (gt? 1 (sum $0))) $0))) $0))"
        dsl = _list_dsl()
        dsl = dsl.add_productions(
            abstraction_production_from_dreamcoder(dsl, inner, inner)
        )
        outer_prod = abstraction_production_from_dreamcoder(dsl, outer, outer)
        self.assertEqual(_render_signature(outer_prod.type_signature()), "[[i]] -> [b]")

    def test_partially_applied_higher_order_argument_asserts(self):
        # mapi_1's slot expects a 2-argument callback (arity 2); supplying only
        # one lambda cannot happen under Stitch's eta_long flag, so assert.
        one_lambda = types.SimpleNamespace(
            symbol="lambda",
            children=(types.SimpleNamespace(symbol="0", children=()),),
        )
        with self.assertRaises(AssertionError):
            _convert_stitch_node(one_lambda, 2, _list_dsl(), {}, [], 0)


def matchBr(s: str, ind: int) -> int | None:
    """
    Given an opening bracket at position ind in string s, find the  position of the corresponding closing bracket.

    Arguments:
    s (str): denoting the solution program expression (already processed by replacements())
    ind (int): is an integer denoting the starting position of the start bracket '('

    Returns:
    int | None: an integer denoting position of closing bracket. If start index does not have an open bracket or no closing brackets close the starting bracket, returns None.
    """
    brPair = 0
    for j in range(ind, len(s)):
        if s[j] == "(":
            brPair += 1
        if s[j] == ")":
            brPair -= 1
        if brPair == 0:
            if j == ind:
                return None
            return j
    return None


def get_argument_list(s: str) -> list[str] | None:
    """
    Returns function body and arguments present in a lambda function serving as an abstraction in DreamCoder's grammar.

    Args:
        s (str): function body

    Returns:
        list[str] | None: lambda function body and arguments present in the function. If there is neither body nor a set of arguments, this returns None.
    """
    start_bracket = s.find("(")
    if start_bracket != -1:
        end_bracket = matchBr(s, start_bracket)
        if end_bracket == None:
            print(f"No closing bracket found for {s}")
            return None
        remaining_arguments = get_argument_list(s[end_bracket + 2 :])
        if remaining_arguments == None:
            return [s[start_bracket + 1 : end_bracket]]
        else:
            return [s[start_bracket + 1 : end_bracket]] + remaining_arguments
    else:
        return s.split(" ")


def parse_abstraction_dc_to_ns(
    abstraction: str, primitive_list: list[str]
) -> ns.InitializedSExpression | ns.AbstractionIndexParameter | None:
    """
    Converts a DreamCoder (Stitch) abstraction or a subabstraction within it to a NeuroSym InitializedSExpression or a AbstractionIndexParameter.

    Example of such a Stitch abstraction string:
    "#(lambda (lambda (#(lambda (mathDomain_swap (mathDomain_simplify (mathDomain_rrotate (mathDomain_swap (mathDomain_div (mathDomain_swap (mathDomain_simplify (mathDomain_rrotate $0 mathDomain_4) mathDomain_0) mathDomain_0) mathDomain_3) mathDomain_5) mathDomain_4) mathDomain_0) mathDomain_0)) (mathDomain_swap (#(lambda (mathDomain_simplify (mathDomain_dist (mathDomain_rrotate (mathDomain_sub $0 mathDomain_5) mathDomain_1) mathDomain_1) mathDomain_0)) (mathDomain_multone (mathDomain_rrotate (#(lambda (mathDomain_simplify (mathDomain_dist (mathDomain_rrotate (mathDomain_sub $0 mathDomain_5) mathDomain_1) mathDomain_1) mathDomain_0)) (mathDomain_swap (#(lambda (mathDomain_swap (mathDomain_simplify $0 mathDomain_0) mathDomain_0)) $0) $1)) mathDomain_4) mathDomain_5)) mathDomain_5))))"

    Example output of such a conversion:
    (lambda (mathDomain_swap (mathDomain_simplify (mathDomain_rrotate (mathDomain_swap (mathDomain_div (mathDomain_swap (mathDomain_simplify (mathDomain_rrotate (mathDomain_swap (mathDomain_simplify (mathDomain_dist (mathDomain_rrotate (mathDomain_sub (mathDomain_multone (mathDomain_rrotate (mathDomain_simplify (mathDomain_dist (mathDomain_rrotate (mathDomain_sub (mathDomain_swap (mathDomain_swap (mathDomain_simplify #0 (mathDomain_0)) (mathDomain_0)) #1) (mathDomain_5)) (mathDomain_1)) (mathDomain_1)) (mathDomain_0)) (mathDomain_4)) (mathDomain_5)) (mathDomain_5)) (mathDomain_1)) (mathDomain_1)) (mathDomain_0)) (mathDomain_5)) (mathDomain_4)) (mathDomain_0)) (mathDomain_0)) (mathDomain_3)) (mathDomain_5)) (mathDomain_4)) (mathDomain_0)) (mathDomain_0)))

    Args:
        abstraction (str): Abstraction discovered by Stitch in DreamCoder

    Returns:
        ns.InitializedSExpression |  ns.AbstractionIndexParameter | None: Abstraction in NeuroSym format. If the abstraction is not parseable, returns None.
    """
    if abstraction == "":
        print("Empty abstraction")
        return None
    if abstraction[0] == "(":
        if matchBr(abstraction, 0) == len(abstraction) - 1:
            return parse_abstraction_dc_to_ns(abstraction[1:-1], primitive_list)
        else:
            print(f"Unmatched brackets in {abstraction}")
            return None
    elif abstraction[0] == "#":
        end_bracket = matchBr(abstraction, 1)
        if end_bracket == len(abstraction) - 1:
            return parse_abstraction_dc_to_ns(abstraction[2:-1], primitive_list)
        else:
            if end_bracket == None:
                print(f"No closing bracket found for {abstraction}")
                return None
            argument_list = get_argument_list(abstraction[1:])
            if argument_list == None:
                print(f"No arguments found for {abstraction}")
                return None
            arg_tuple = tuple(
                [
                    parse_abstraction_dc_to_ns(arg_abstraction, primitive_list)
                    for arg_abstraction in argument_list
                ]
            )
            if None in arg_tuple:
                print(f"None in arg_tuple for {abstraction}")
            arg_tuple = tuple([x for x in arg_tuple if x != None])
            if len(arg_tuple) == 0:
                return None
            if len(arg_tuple) == 1:
                return arg_tuple[0]
            sub_abstraction_body = arg_tuple[0]
            sub_abstraction_args = arg_tuple[1:]
            # This assert ensures that the function at the root of the InitializedSExpression has the correct types
            assert isinstance(sub_abstraction_body, ns.InitializedSExpression)
            result = _with_index_parameters(
                sub_abstraction_body, sub_abstraction_args, True
            )
            # This assert ensures that the final result of calling _with_index_parameters has the correct types and not just an "object" type (derived from the apply function call in the _with_index_parameters function)
            assert isinstance(result, ns.InitializedSExpression) or isinstance(
                result, ns.AbstractionIndexParameter
            )
            return result
            # return ns.InitializedSExpression("lambda", arg_tuple, {})
    elif abstraction.split(" ")[0] == "lambda":
        end_bracket = matchBr(abstraction, 7)
        if end_bracket == len(abstraction) - 1:
            return parse_abstraction_dc_to_ns(abstraction[8:-1], primitive_list)
        else:
            if end_bracket == None:
                print(f"No closing bracket found for {abstraction}")
                return None
            argument_list = get_argument_list(abstraction[7:end_bracket])
            if argument_list == None:
                print(f"No arguments found for {abstraction}")
                return None
            arg_tuple = tuple(
                [
                    parse_abstraction_dc_to_ns(arg_abstraction, primitive_list)
                    for arg_abstraction in argument_list
                ]
            )
            if None in arg_tuple:
                print(f"None in arg_tuple for {abstraction}")
            arg_tuple = tuple([x for x in arg_tuple if x != None])
            return ns.InitializedSExpression("lambda", arg_tuple, {})
    else:
        splits = abstraction.split(" ")
        func = splits[0]
        if func in primitive_list:
            argument_list = get_argument_list(abstraction[len(func) + 1 :])
            if argument_list == None:
                print(f"No arguments found for {abstraction}")
                return None
            arg_tuple = tuple(
                [
                    parse_abstraction_dc_to_ns(arg_abstraction, primitive_list)
                    for arg_abstraction in argument_list
                ]
            )
            if None in arg_tuple:
                print(f"None in arg_tuple for {abstraction}")
            arg_tuple = tuple([x for x in arg_tuple if x != None])
            return ns.InitializedSExpression(func, arg_tuple, {})
        else:
            if abstraction[0] == "$":
                abstraction_child = ns.AbstractionIndexParameter(
                    int(abstraction.split(" ")[0][1:])
                )
                return abstraction_child
            else:
                return ns.InitializedSExpression(abstraction, (), {})


def enumerate_dsl(family, dist, min_likelihood=-6, max_denominator=10**6, count=None):
    result = list(
        itertools.islice(family.enumerate(dist, min_likelihood=min_likelihood), count)
    )
    result = [
        (
            ns.render_s_expression(prog),
            Fraction(*np.exp(likelihood).as_integer_ratio()).limit_denominator(
                max_denominator=max_denominator
            ),
        )
        for prog, likelihood in result
    ]
    result = sorted(result, key=lambda x: (-x[1], ns.render_s_expression(x[0])))
    result_display = str(result)
    print("{" + result_display[1:-1] + "}")
    return set(result)


if __name__ == "__main__":
    s_exp = ns.SExpression(
        "lambda",
        [
            parse_abstraction_dc_to_ns(
                "#(lambda (lambda (#(lambda (mathDomain_swap (mathDomain_simplify (mathDomain_rrotate (mathDomain_swap (mathDomain_div (mathDomain_swap (mathDomain_simplify (mathDomain_rrotate $0 mathDomain_4) mathDomain_0) mathDomain_0) mathDomain_3) mathDomain_5) mathDomain_4) mathDomain_0) mathDomain_0)) (mathDomain_swap (#(lambda (mathDomain_simplify (mathDomain_dist (mathDomain_rrotate (mathDomain_sub $0 mathDomain_5) mathDomain_1) mathDomain_1) mathDomain_0)) (mathDomain_multone (mathDomain_rrotate (#(lambda (mathDomain_simplify (mathDomain_dist (mathDomain_rrotate (mathDomain_sub $0 mathDomain_5) mathDomain_1) mathDomain_1) mathDomain_0)) (mathDomain_swap (#(lambda (mathDomain_swap (mathDomain_simplify $0 mathDomain_0) mathDomain_0)) $0) $1)) mathDomain_4) mathDomain_5)) mathDomain_5))))",
                [
                    "mathDomain_swap",
                    "mathDomain_simplify",
                    "mathDomain_rrotate",
                    "mathDomain_div",
                    "mathDomain_dist",
                    "mathDomain_sub",
                    "mathDomain_multone",
                    "mathDomain_0",
                    "mathDomain_1",
                    "mathDomain_3",
                    "mathDomain_4",
                    "mathDomain_5",
                ],
            )
        ],
    )
    dslf = ns.DSLFactory()
    dslf.concrete("mathDomain_swap", "(s, i) -> s", lambda st, x: st + "swap" + str(x))
    dslf.concrete(
        "mathDomain_simplify", "(s, i) -> s", lambda st, x: st + "simplify" + str(x)
    )
    dslf.concrete(
        "mathDomain_rrotate", "(s, i) -> s", lambda st, x: st + "rrotate" + str(x)
    )
    dslf.concrete("mathDomain_div", "(s, i) -> s", lambda st, x: st + "div" + str(x))
    dslf.concrete("mathDomain_dist", "(s, i) -> s", lambda st, x: st + "dist" + str(x))
    dslf.concrete("mathDomain_sub", "(s, i) -> s", lambda st, x: st + "sub" + str(x))
    dslf.concrete(
        "mathDomain_multone", "(s, i) -> s", lambda st, x: st + "multone" + str(x)
    )
    dslf.concrete("mathDomain_0", "() -> i", lambda: 0)
    dslf.concrete("mathDomain_1", "() -> i", lambda: 1)
    dslf.concrete("mathDomain_3", "() -> i", lambda: 3)
    dslf.concrete("mathDomain_4", "() -> i", lambda: 4)
    dslf.concrete("mathDomain_5", "() -> i", lambda: 5)
    dslf.lambdas()
    dslf.prune_to("s -> s")
    dsl = dslf.finalize()
    render = ns.render_s_expression(s_exp)
    print(f"\n Final Render: {render} \n")
    # print(f"DSL Productions: {dsl.productions}")
    type_argument = [dsl.compute_type_abs(x) for x in s_exp.children][0]
    # corrected_type_argument = None
    reversed_order_items = [value for _, value in type_argument.env._elements.items()]
    new_type_env_dict = {}
    for key in type_argument.env._elements.keys():
        new_type_env_dict[key] = reversed_order_items.pop()
    corrected_type_argument = TypeWithEnvironment(
        typ=type_argument.typ, env=Environment(_elements=new_type_env_dict)
    )
    print(f"\n {corrected_type_argument}")
    type_signature = FunctionTypeSignature(
        [x[0] for _, x in corrected_type_argument.env._elements.items()],
        corrected_type_argument.typ,
    )
    final = AbstractionProduction("abs_0", type_signature, s_exp)
    dsl = dsl.add_production(final)
    dsl = dsl.with_valid_root_types([parse_type("s -> s")])
    print(dsl._production_by_symbol)
    print(dsl.render())
    family = BigramProgramDistributionFamily(dsl)
    enumerations = list(
        enumerate_dsl(family, family.uniform(), min_likelihood=-8000, count=30)
    )
    # type_signature = FunctionTypeSignature([x.typ for x in type_arguments], type_out)
    # abstraction = AbstractionProduction(abstr_name, type_signature, abstr_body)
    # dsl.add_production(abstraction)
