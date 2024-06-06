import copy

from neurosym.programs.s_expression import SExpression
from neurosym.python_dsl.names import PYTHON_DSL_SEPARATOR
from neurosym.python_dsl.python_ast_tools import clean_type, is_sequence


def run_dfa_on_program(dfa, node, state):
    """
    Runs the dfa on a program, yielding the node and the state
        for each node in the program.

    Args:
        dfa: The dfa.
        node: The node.
        state: The root state

    Yields:
        A tuple of the node and the state, for each node in the program.
    """
    if not isinstance(node, (SExpression, str)):
        raise ValueError(f"expected SExpression or str, got {node}")
    yield node, state
    if not isinstance(node, SExpression):
        return
    if not node.children:
        # avoid looking up dfa[state][tag] when there are no children; allows sparser dfa
        return
    if state not in dfa:
        raise ValueError(f"state {state} not in dfa")
    if node.symbol not in dfa[state]:
        raise ValueError(f"symbol {node.symbol} not in dfa[{state}]")
    dfa_states = dfa[state][node.symbol]
    for i, child in enumerate(node.children):
        yield from run_dfa_on_program(dfa, child, dfa_states[i % len(dfa_states)])


def add_disambiguating_type_tags(dfa, prog, start_state):
    """
    Add disambiguating type tags to a program, which appended to each symbol in the program,
        after the separator. Also adds a sequence length tag if the symbol is a sequence type.

    Args:
        dfa: The dfa.
        prog: The program.
        start_state: The state for the root node.

    Returns:
        The program with disambiguating type tags.
    """
    prog = copy.deepcopy(prog)
    node_id_to_new_symbol = {}
    for node, tag in run_dfa_on_program(dfa, prog, start_state):
        assert isinstance(node, SExpression), node
        new_symbol = node.symbol + PYTHON_DSL_SEPARATOR + clean_type(tag)
        if is_sequence(tag, node.symbol):
            new_symbol += PYTHON_DSL_SEPARATOR + str(len(node.children))
        node_id_to_new_symbol[id(node)] = new_symbol
    return prog.replace_symbols_by_id(node_id_to_new_symbol)
