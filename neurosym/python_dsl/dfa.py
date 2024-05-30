import ast
from typing import Dict, List

from frozendict import frozendict

from .python_ast_tools import fields_for_node

# exclude these tags from the dfa. these are all python 3.10+ features,
# and for consistency across python versions, we exclude them. We can
# add them back in later if we want to support them.
excluded_python_tags = [
    # match
    "Match",
    "MatchAs",
    "MatchMapping",
    "MatchSequence",
    "MatchValue",
    "MatchClass",
    "MatchOr",
    "MatchSingleton",
    "MatchStar",
    "match_case",
    "pattern",
    # trystar
    "TryStar",
]


default_transition_dict = frozendict(
    {
        "M": {ast.Module: {"body": "seqS", "type_ignores": "[TI]"}},
        "S": {
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef): {
                "body": "seqS",
                "decorator_list": "[E]",
                "bases": "[E]",
                "name": "Name",
                "args": "As",
                "returns": "TA",
                "type_comment": "TC",
                "keywords": "[K]",
            },
            ast.Return: {"value": "E"},
            ast.Delete: {"targets": "[L]"},
            (ast.Assign, ast.AugAssign, ast.AnnAssign): {
                "value": "E",
                "targets": "[L]",
                "target": "L",
                "type_comment": "TC",
                "op": "O",
                "annotation": "TA",
                "simple": "bool",
            },
            (
                ast.For,
                ast.AsyncFor,
                ast.While,
                ast.If,
                ast.With,
                ast.AsyncWith,
                ast.Try,
            ): {
                "iter": "E",
                "test": "E",
                "body": "seqS",
                "orelse": "seqS",
                "finalbody": "seqS",
                "items": "[W]",
                "handlers": "[EH]",
                "target": "L",
                "type_comment": "TC",
            },
            #         ast.Match: {"subject": "E", "cases": "C"},
            ast.Raise: {all: "E"},
            ast.Assert: {all: "E"},
            (ast.Import, ast.ImportFrom): {
                "names": "[alias]",
                "module": "NullableNameStr",
                "level": "int",
            },
            (ast.Global, ast.Nonlocal): {"names": "[NameStr]"},
            ast.Expr: {"value": "E"},
        },
        "E": {
            (ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare): {
                "op": "O",
                "ops": "[O]",
                "comparators": "[E]",
                "values": "[E]",
                all: "E",
            },
            ast.NamedExpr: {"value": "E", "target": "L"},
            ast.Lambda: {"args": "As", "body": "E"},
            (
                ast.IfExp,
                ast.Dict,
                ast.Set,
                ast.List,
                ast.Tuple,
                ast.Await,
                ast.Yield,
                ast.YieldFrom,
            ): {
                "ctx": "Ctx",
                "elts": "[StarredRoot]",
                "keys": "[E]",
                "values": "[E]",
                all: "E",
            },
            (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp): {
                "generators": "[C]",
                all: "E",
            },
            ast.Call: {
                "keywords": "[K]",
                "args": "[StarredRoot]",
                all: "E",
            },
            ast.JoinedStr: {"values": "[F]"},
            ast.Constant: {"value": "Const", "kind": "ConstKind"},
            ast.Name: {"id": "Name", "ctx": "Ctx"},
            (ast.Attribute, ast.Subscript, ast.Starred): {
                "value": "E",
                "attr": "NameStr",
                "slice": "SliceRoot",
                "ctx": "Ctx",
            },
        },
        "SliceRoot": {
            "_slice_content": {all: "E"},
            "_slice_slice": {all: "Slice"},
            "_slice_tuple": {all: "SliceTuple"},
        },
        "SliceTuple": {
            ast.Tuple: {"elts": "[SliceRoot]", "ctx": "Ctx"},
        },
        "[SliceRoot]": {"list": "SliceRoot"},
        "StarredRoot": {
            "_starred_content": {all: "E"},
            "_starred_starred": {all: "Starred"},
        },
        "Starred": {ast.Starred: {"value": "E", "ctx": "Ctx"}},
        "Slice": {
            ast.Slice: {"lower": "E", "upper": "E", "step": "E"},
        },
        "As": {
            ast.arguments: {
                ("kw_defaults", "defaults"): "[E]",
                ("args", "posonlyargs", "kwonlyargs"): "[A]",
                ("vararg", "kwarg"): "A",
            }
        },
        "A": {
            ast.arg: {"annotation": "TA", "arg": "Name", "type_comment": "TC"},
        },
        "F": {
            ast.FormattedValue: {"value": "E", "format_spec": "F", "conversion": "int"},
            ast.Constant: {all: "F"},
            ast.JoinedStr: {"values": "[F]"},
        },
        "C": {
            ast.comprehension: {
                "target": "L",
                "iter": "E",
                "ifs": "[E]",
                "is_async": "bool",
            }
        },
        "K": {ast.keyword: {"value": "E", "arg": "NullableNameStr"}},
        "EH": {
            ast.ExceptHandler: {
                "type": "E",
                "name": "NullableName",
                "body": "seqS",
            }
        },
        "W": {
            ast.withitem: {
                "context_expr": "E",
                "optional_vars": "L",
            }
        },
        "L": {
            ast.Name: {
                "id": "Name",
                "ctx": "Ctx",
            },
            ast.Tuple: {"elts": "[L]", "ctx": "Ctx"},
            ast.List: {"elts": "[L]", "ctx": "Ctx"},
            ast.Subscript: {"value": "E", "slice": "SliceRoot", "ctx": "Ctx"},
            ast.Attribute: {"value": "E", "attr": "NameStr", "ctx": "Ctx"},
            ast.Starred: {"value": "L", "ctx": "Ctx"},
            "_starred_content": {all: "L"},
            "_starred_starred": {all: "L"},
        },
        "seqS": {},
        "[E]": {"list": "E"},
        "[StarredRoot]": {"list": "StarredRoot"},
        "alias": {
            ast.alias: {
                "name": "NameStr",
                "asname": "NullableNameStr",
            },
        },
        "[NameStr]": {"list": "NameStr"},
        "TA": {all: {all: "TA"}, "list": "TA"},
        "[F]": {"list": "F"},
        "[A]": {"list": "A"},
        "[C]": {"list": "C"},
        "[EH]": {"list": "EH"},
        "[K]": {"list": "K"},
        "[L]": {"list": "L"},
        "[O]": {"list": "O"},
        "[W]": {"list": "W"},
        "[alias]": {"list": "alias"},
        "[TI]": {"list": "TI"},
    }
)


def python_dfa(
    transitions=default_transition_dict,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Export a Discrete Tree Finite Automaton for the Python AST,
        in the form of a dictionary of the form dict[state, dict[tag, list[state]]].
    """
    all_tags = [
        x
        for x in dir(ast)
        if isinstance(getattr(ast, x), type)
        and issubclass(getattr(ast, x), ast.AST)
        and x not in excluded_python_tags
    ]

    extras = [
        "_slice_content",
        "_slice_slice",
        "_slice_tuple",
        "_starred_content",
        "_starred_starred",
    ]

    result = {}
    for state in transitions:
        result[state] = {}
        for tag in all_tags:
            t = getattr(ast, tag)
            out = compute_transition(transitions, state, t, fields_for_node(t))
            if out is not None:
                result[state][tag] = out
        for tag in extras:
            out = compute_transition(transitions, state, tag, [None])
            if out is not None:
                result[state][tag] = out

        missing = (
            set(all_types_as_string(list(transitions[state])))
            - set(result[state])
            - {"list", "/seq", "/splice"}
        )
        if missing:
            raise RuntimeError(f"in state {state}: missing {missing}")

        if "list" in transitions[state]:
            result[state]["list"] = [transitions[state]["list"]]
    result["seqS"]["/seq"] = ["S"]
    result["seqS"]["/subseq"] = ["S"]
    result["seqS"]["/choiceseq"] = ["S"]
    result["S"]["/splice"] = ["seqS"]
    return result


def compute_transition(transitions, state, typ, fields):
    """
    Compute the list of states that the DFA should transition to
        for each child of a node of type `typ` in state `state`.

    :param transitions: the transition dictionary
    :param state: the current state
    :param typ: the type of the node
    :param fields: the fields of the node

    :return: a list of states to transition to, or None if
        the node could not be matched.
    """
    transition = transitions[state]
    transition = compute_match(transition, typ, default=False)
    if transition is False:
        return None
    return [compute_match(transition, field) for field in fields]


def all_types_as_string(ts):
    """
    Converts all the types in the given set into strings.
        E.g., (ast.Name, ast.If) -> ["Name", "If"]
    """
    if isinstance(ts, (list, tuple)):
        for x in ts:
            yield from all_types_as_string(x)
        return
    if ts is all:
        return
    if isinstance(ts, type):
        yield ts.__name__
        return
    assert isinstance(ts, str), ts
    yield ts


def compute_match(transition, key, default=None):
    """
    Compute the match for the given key in the transition dictionary. Handles tuples
        and the special case where the key is `all`.

    :param transition: the transition dictionary
    :param key: the key to look up
    :param default: the default value to return if the key is not found

    :return: the value corresponding to the key in the transition dictionary
    """
    for k, v in transition.items():
        if k is all:
            return v
        if not isinstance(k, tuple):
            k = (k,)
        if key in k:
            return v
    if default is not None:
        return default
    raise RuntimeError(f"could not find {key} in {transition}")
