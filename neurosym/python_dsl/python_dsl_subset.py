from collections import defaultdict
from dataclasses import dataclass, field
from types import NoneType
from typing import Callable, Dict, List, Set, Tuple, Union

from neurosym.dsl.dsl_factory import DSLFactory
from neurosym.programs.s_expression import SExpression
from neurosym.python_dsl import python_ast_tools
from neurosym.python_dsl.convert_python.convert import to_type_annotated_ns_s_exp
from neurosym.python_dsl.convert_python.python_ast import PythonAST
from neurosym.types.type import ArrowType
from neurosym.types.type_string_repr import parse_type, render_type

from .names import PYTHON_DSL_SEPARATOR


@dataclass
class PythonDSLSubset:
    """
    Represents a subset of the python DSL. This is represented as
        - a dictionary from sequential type to a list of lengths.
        - a dictionary from types to a list of leaves of that type
    """

    _lengths_by_sequence_type: Dict[str, Set[int]] = field(
        default_factory=lambda: defaultdict(set)
    )
    _leaves: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    @property
    def lengths_by_sequence_type(self) -> Dict[str, List[int]]:
        """
        Compute the lengths by sequence type.
        """
        return {k: sorted(v) for k, v in self._lengths_by_sequence_type.items()}

    @property
    def leaves(self) -> Dict[str, List[str]]:
        """
        Compute the leaves.
        """
        return {k: sorted(v) for k, v in self._leaves.items()}

    def add_s_exps(self, *s_exps):
        """
        Add the following s-expressions to the subset. They must be type-annotated.

        :param s_exps: the s-expressions to add
        """
        for s_exp in s_exps:
            for node in _traverse(s_exp):
                assert isinstance(node, SExpression)
                symbol, state, *_ = node.symbol.split(PYTHON_DSL_SEPARATOR)
                state = python_ast_tools.unclean_type(state)
                if python_ast_tools.is_sequence(state, symbol):
                    self._lengths_by_sequence_type[state].add(len(node.children))
                elif len(node.children) == 0:
                    self._leaves[state].add(symbol)

    @classmethod
    def from_s_exps(cls, s_exps):
        """
        Factory version of ``add_s_exps``.

        :param s_exps: the s-expressions to add
        """
        subset = cls()
        subset.add_s_exps(*s_exps)
        return subset

    def add_programs(
        self,
        dfa,
        *programs: Tuple[PythonAST, ...],
        root: Union[str, Tuple[str, ...]],
    ):
        """
        Add the programs to the subset. The root symbol of the DSL is passed as an argument,
            and can be a single string or a tuple of strings.

        :param dfa: the dfa of the DSL
        :param programs: the programs to extract the subset from
        :param root: the root symbol of the DSL. If a tuple is passed, it must
            be the same length as the programs, providing a root symbol for each program.
        :param abstrs: abstractions: their bodies will be added to the list of programs
        """
        if isinstance(root, str):
            root = [root] * len(programs)
        else:
            if len(root) != len(programs):
                raise ValueError(
                    "The length of the root should be the same as the number of programs, but was"
                    f" {len(root)} and {len(programs)} respectively."
                )

        s_exps = []
        for program, root_sym in zip(programs, root):
            s_exp = to_type_annotated_ns_s_exp(program, dfa, root_sym)
            self.add_s_exps(s_exp)
            s_exps.append(s_exp)
        return s_exps

    @classmethod
    def from_programs(
        cls, dfa, *programs: Tuple[PythonAST, ...], root: Union[str, Tuple[str, ...]]
    ):
        """
        Factory version of ``add_programs``.
        """
        subset = cls()
        subset.add_programs(dfa, *programs, root=root)
        return subset

    def fill_in_missing_lengths(self):
        """
        Fill in "missing lengths" for each sequence type. E.g., if the lengths
        of a sequence type are [1, 3], this function will add 2 to the list.
        """
        self._lengths_by_sequence_type = {
            seq_type: set(range(min(lengths), max(lengths) + 1))
            for seq_type, lengths in self.lengths_by_sequence_type.items()
        }


def _traverse(s_exp):
    """
    Yield all the nodes in the s-expression.
    """
    yield s_exp
    for child in s_exp.children:
        yield from _traverse(child)


def create_python_dsl(
    dfa: dict,
    dsl_subset: PythonDSLSubset,
    start_state: str,
    add_additional_productions: Callable[[DSLFactory], NoneType] = lambda dslf: None,
):
    """
    Create a DSL from a DFA and a subset of the DSL.

    :param dfa: the DFA of the DSL. See ``ns.python_dfa()`` for more information.
    :param dsl_subset: the subset of the DSL
    :param start_state: the root state of the DSL
    :param add_additional_productions: a function that adds additional productions to the DSL
    """
    dslf = DSLFactory()
    for target in dfa:
        for prod in dfa[target]:
            input_types = tuple(parse_type(t) for t in dfa[target][prod])
            if python_ast_tools.is_sequence(target, prod):
                assert len(input_types) == 1
                for length in dsl_subset.lengths_by_sequence_type.get(target, []):
                    typ = ArrowType(input_types * length, parse_type(target))
                    dslf.concrete(
                        prod
                        + PYTHON_DSL_SEPARATOR
                        + python_ast_tools.clean_type(target)
                        + PYTHON_DSL_SEPARATOR
                        + str(length),
                        render_type(typ),
                        None,
                    )
            else:
                typ = ArrowType(tuple(input_types), parse_type(target))
                dslf.concrete(
                    prod + PYTHON_DSL_SEPARATOR + python_ast_tools.clean_type(target),
                    render_type(typ),
                    None,
                )
    for target, leaves in dsl_subset.leaves.items():
        for constant in leaves:
            typ = ArrowType((), parse_type(target))
            dslf.concrete(
                constant + PYTHON_DSL_SEPARATOR + target, render_type(typ), None
            )
    add_additional_productions(dslf)
    dslf.prune_to(start_state, tolerate_pruning_entire_productions=True)
    return dslf.finalize()
