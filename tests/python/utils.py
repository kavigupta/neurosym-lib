import ast
import json
from functools import lru_cache
from textwrap import dedent

import neurosym as ns


def fit_to(
    programs,
    parser=ns.python_to_python_ast,
    root="M",
    use_def_use=True,
    use_node_ordering=True,
    include_type_preorder_mask=True,
):
    """
    Set include_type_preorder_mask to False to disable the type preorder mask,
        this is basically only useful in the specific context where we are testing
        the names mask and want no other masks to be applied.
    """
    dfa = ns.python_dfa()
    programs = [parser(p) for p in programs]
    subset = ns.PythonDSLSubset.from_programs(dfa, *programs, root=root)
    dsl = ns.create_python_dsl(dfa, subset, root)
    apms = [
        lambda dist, dsl: ns.python_def_use_mask.DefUseChainPreorderMask(
            dist, dsl, ns.python_def_use_mask.DefUseMaskConfiguration(dfa, {})
        )
    ]
    node_ordering = (
        ns.python_def_use_mask.PythonNodeOrdering
        if use_node_ordering
        else ns.DefaultNodeOrdering
    )
    fam = ns.BigramProgramDistributionFamily(
        dsl,
        additional_preorder_masks=apms if use_def_use else [],
        include_type_preorder_mask=include_type_preorder_mask,
        node_ordering=node_ordering,
    )
    counts = fam.count_programs(
        [[ns.to_type_annotated_ns_s_exp(program, dfa, root) for program in programs]]
    )
    dist = fam.counts_to_distribution(counts)[0]
    return dfa, dsl, fam, dist


@lru_cache(None)
def small_set_runnable_code_examples():
    with open("test_data/small_set_runnable_code.json") as f:
        contents = json.load(f)
    return contents


def cwq(s):
    """
    Canonicalize with question marks and dollars
    """
    s = dedent(s)
    s = s.replace("?", "__QUESTION__MARK__")
    s = s.replace("$", "__DOLLAR__")
    s = ast.unparse(ast.parse(s))
    s = s.replace("__QUESTION__MARK__", "?")
    s = s.replace("__DOLLAR__", "$")
    return s
