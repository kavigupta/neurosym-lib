import neurosym as ns
from neurosym.dsl.abstraction import _with_index_parameters
from neurosym.dsl.abstraction import AbstractionProduction
from neurosym.types.type_signature import FunctionTypeSignature
from neurosym.types.type_with_environment import Environment, TypeWithEnvironment
from neurosym.types.type import AtomicType
from neurosym.program_dist.bigram import BigramProgramDistributionFamily
from fractions import Fraction
import numpy as np

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
        if s[j]=="(":
            brPair+=1
        if s[j]==")":
            brPair-=1
        if brPair==0:
            if j==ind:
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
        remaining_arguments = get_argument_list(s[end_bracket+2:])
        if remaining_arguments == None:
            return [s[start_bracket+1:end_bracket]]
        else:
            return [s[start_bracket+1:end_bracket]] + remaining_arguments
    else:
        return s.split(" ")
    
def parse_abstraction_dc_to_ns(abstraction: str, primitive_list: list[str]) -> ns.InitializedSExpression | ns.AbstractionIndexParameter | None:
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
        if matchBr(abstraction, 0) == len(abstraction)-1:
            return parse_abstraction_dc_to_ns(abstraction[1:-1], primitive_list)
        else:
            print(f"Unmatched brackets in {abstraction}")
            return None
    elif abstraction[0] == "#":
        end_bracket = matchBr(abstraction, 1)
        if end_bracket == len(abstraction)-1:
            return parse_abstraction_dc_to_ns(abstraction[2:-1], primitive_list)
        else:
            if end_bracket == None:
                print(f"No closing bracket found for {abstraction}")
                return None
            argument_list = get_argument_list(abstraction[1:])
            if argument_list == None:
                print(f"No arguments found for {abstraction}")
                return None
            arg_tuple = tuple([parse_abstraction_dc_to_ns(arg_abstraction, primitive_list) for arg_abstraction in argument_list])
            if None in arg_tuple:
                print(f"None in arg_tuple for {abstraction}")
            arg_tuple = tuple([x for x in arg_tuple if x != None])
            if len(arg_tuple) == 0:
                return None
            if len(arg_tuple) == 1:
                return arg_tuple[0]
            sub_abstraction_body = arg_tuple[0]
            sub_abstraction_args = arg_tuple[1:]
            #This assert ensures that the function at the root of the InitializedSExpression has the correct types
            assert isinstance(sub_abstraction_body, ns.InitializedSExpression)
            result = _with_index_parameters(sub_abstraction_body, sub_abstraction_args, True)
            #This assert ensures that the final result of calling _with_index_parameters has the correct types and not just an "object" type (derived from the apply function call in the _with_index_parameters function)
            assert isinstance(result, ns.InitializedSExpression) or isinstance(result, ns.AbstractionIndexParameter)
            return result
            # return ns.InitializedSExpression("lambda", arg_tuple, {})
    elif abstraction.split(" ")[0] == "lambda":
        end_bracket = matchBr(abstraction, 7)
        if end_bracket  == len(abstraction)-1:
            return parse_abstraction_dc_to_ns(abstraction[8:-1], primitive_list)
        else:
            if end_bracket == None:
                print(f"No closing bracket found for {abstraction}")
                return None
            argument_list = get_argument_list(abstraction[7:end_bracket])
            if argument_list == None:
                print(f"No arguments found for {abstraction}")
                return None
            arg_tuple = tuple([parse_abstraction_dc_to_ns(arg_abstraction, primitive_list) for arg_abstraction in argument_list])
            if None in arg_tuple:
                print(f"None in arg_tuple for {abstraction}")
            arg_tuple = tuple([x for x in arg_tuple if x != None])
            return ns.InitializedSExpression("lambda", arg_tuple, {})
    else:
        splits = abstraction.split(" ")
        func = splits[0]
        if func in primitive_list:
            argument_list = get_argument_list(abstraction[len(func)+1:])
            if argument_list == None:
                print(f"No arguments found for {abstraction}")
                return None
            arg_tuple = tuple([parse_abstraction_dc_to_ns(arg_abstraction, primitive_list) for arg_abstraction in argument_list])
            if None in arg_tuple:
                print(f"None in arg_tuple for {abstraction}")
            arg_tuple = tuple([x for x in arg_tuple if x != None])
            return ns.InitializedSExpression(func, arg_tuple, {})
        else:
            if abstraction[0] == "$":
                abstraction_child = ns.AbstractionIndexParameter(int(abstraction.split(" ")[0][1:]))
                return abstraction_child
            else:
                return ns.InitializedSExpression(abstraction, (), {})

def enumerate_dsl(family, dist, min_likelihood=-6, max_denominator=10**6):
    result = list(family.enumerate(dist, min_likelihood=min_likelihood))
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
    s_exp = ns.SExpression("lambda", [parse_abstraction_dc_to_ns("#(lambda (lambda (#(lambda (mathDomain_swap (mathDomain_simplify (mathDomain_rrotate (mathDomain_swap (mathDomain_div (mathDomain_swap (mathDomain_simplify (mathDomain_rrotate $0 mathDomain_4) mathDomain_0) mathDomain_0) mathDomain_3) mathDomain_5) mathDomain_4) mathDomain_0) mathDomain_0)) (mathDomain_swap (#(lambda (mathDomain_simplify (mathDomain_dist (mathDomain_rrotate (mathDomain_sub $0 mathDomain_5) mathDomain_1) mathDomain_1) mathDomain_0)) (mathDomain_multone (mathDomain_rrotate (#(lambda (mathDomain_simplify (mathDomain_dist (mathDomain_rrotate (mathDomain_sub $0 mathDomain_5) mathDomain_1) mathDomain_1) mathDomain_0)) (mathDomain_swap (#(lambda (mathDomain_swap (mathDomain_simplify $0 mathDomain_0) mathDomain_0)) $0) $1)) mathDomain_4) mathDomain_5)) mathDomain_5))))", ["mathDomain_swap", "mathDomain_simplify", "mathDomain_rrotate", "mathDomain_div", "mathDomain_dist", "mathDomain_sub", "mathDomain_multone", "mathDomain_0", "mathDomain_1", "mathDomain_3", "mathDomain_4", "mathDomain_5"])])
    dslf = ns.DSLFactory()
    dslf.concrete("mathDomain_swap", "(s, i) -> s", lambda st, x: st + "swap" + str(x))
    dslf.concrete("mathDomain_simplify", "(s, i) -> s", lambda st, x: st + "simplify" + str(x))
    dslf.concrete("mathDomain_rrotate", "(s, i) -> s", lambda st, x: st + "rrotate" + str(x))
    dslf.concrete("mathDomain_div", "(s, i) -> s", lambda st, x: st + "div" + str(x))
    dslf.concrete("mathDomain_dist", "(s, i) -> s", lambda st, x: st + "dist" + str(x))
    dslf.concrete("mathDomain_sub", "(s, i) -> s", lambda st, x: st + "sub" + str(x))
    dslf.concrete("mathDomain_multone", "(s, i) -> s", lambda st, x: st + "multone" + str(x))
    dslf.concrete("mathDomain_0", "() -> i", lambda: 0)
    dslf.concrete("mathDomain_1", "() -> i", lambda: 1)
    dslf.concrete("mathDomain_3", "() -> i", lambda: 3)
    dslf.concrete("mathDomain_4", "() -> i", lambda: 4)
    dslf.concrete("mathDomain_5", "() -> i", lambda: 5)
    dsl = dslf.finalize()
    render = ns.render_s_expression(s_exp)
    print(f"\n Final Render: {render} \n")
    #print(f"DSL Productions: {dsl.productions}")
    type_argument = [dsl.compute_type_abs(x) for x in s_exp.children][0]
    #corrected_type_argument = None
    reversed_order_items = [value for _, value in type_argument.env._elements.items()]
    new_type_env_dict = {}
    for key in type_argument.env._elements.keys():
        new_type_env_dict[key] = reversed_order_items.pop()
    corrected_type_argument = TypeWithEnvironment(typ = type_argument.typ, env = Environment(_elements = new_type_env_dict))
    print(f"\n {corrected_type_argument}")
    type_signature = FunctionTypeSignature([x[0] for _, x in corrected_type_argument.env._elements.items()], corrected_type_argument.typ)
    final = AbstractionProduction(render, type_signature, s_exp)
    dsl = dsl.add_production(final)
    dsl = dsl.with_valid_root_types([AtomicType("s")]) 
    family = BigramProgramDistributionFamily(dsl)
    enumerations = list(enumerate_dsl(family, family.uniform(), min_likelihood=-800)) 
    #type_signature = FunctionTypeSignature([x.typ for x in type_arguments], type_out)
    #abstraction = AbstractionProduction(abstr_name, type_signature, abstr_body) 
    #dsl.add_production(abstraction)