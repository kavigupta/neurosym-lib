from dataclasses import dataclass
from typing import Callable, Dict, List, Set, Tuple, Union

from torch import nn

from neurosym.programs.s_expression_render import symbols_for_program
from neurosym.search_graph.dsl_search_node import DSLSearchNode
from neurosym.types.type_signature import FunctionTypeSignature
from neurosym.utils.documentation import internal_only

from ...dsl.dsl import DSL
from ...dsl.production import ParameterizedProduction, Production
from ...programs.hole import Hole
from ...programs.s_expression import InitializedSExpression, SExpression
from ...types.type import ArrowType, ListType, TensorType, Type


class PartialProgramNotFoundError(Exception):
    """
    Raised when a partial program cannot be found for a hole.
    """


@dataclass
class NeuralDSL(DSL):
    """
    A neural DSL extends ``DSL`` to handle neural heuristics (ie: type-appropriate NN productions)
    These neural heuristics can be used to fill holes in partial programs.
    """

    # partial_programs: Dict[Type, SExpression]
    type_to_symbol: Dict[Type, str]
    original_symbols: Set[str]
    semantics: Dict[Type, Callable]
    initializers: Dict[str, Callable[[], nn.Module]]

    @classmethod
    def from_dsl(
        cls, dsl: DSL, modules: Dict[Type, Tuple[str, Callable[[], nn.Module]]]
    ):
        """
        Creates a NeuralDSL from a DSL and a set of type specific modules.

        The type specific modules are used to fill holes in partial programs.

        :param dsl: The DSL to extend.
        :param modules: A dictionary mapping types to tags and functions that
            are used to initialize the modules for that type.
        """
        type_to_symbol = {}

        semantics = {}
        initializers = {}
        for fn_type, (_, module_template) in modules.items():
            semantics[fn_type] = _inject_environment_argument(fn_type)
            initializers[fn_type] = dict(initialized_module=module_template)

        productions = dsl.productions

        return cls(
            productions=productions,
            valid_root_types=dsl.valid_root_types,
            max_type_depth=dsl.max_type_depth,
            max_env_depth=dsl.max_env_depth,
            type_to_symbol=type_to_symbol,
            original_symbols=set(dsl.symbols()),
            semantics=semantics,
            initializers=initializers,
        )

    def initialize(self, program: SExpression) -> InitializedSExpression:
        """
        Initializes all the productions in the given program.

        Returns a new program with the same structure, but with all the productions
        initialized.
        """
        if isinstance(program, Hole):
            try:
                initialized = {k: v() for k, v in self.initializers[program.twe.typ].items()}
                return _NeuralHole(initialized, self.semantics[program.twe.typ])

            except KeyError as e:
                raise PartialProgramNotFoundError(
                    f"Cannot initialize program {program}."
                ) from e
        return super().initialize(program)

    def program_has_no_holes(self, program: Union[SExpression, DSLSearchNode]) -> bool:
        """
        Returns True if the given program has no holes.
        """
        if isinstance(program, DSLSearchNode):
            program = program.program
        assert isinstance(program, SExpression)
        return symbols_for_program(program) - self.original_symbols == set()


class _NeuralHole:
    """
    A hole that can be filled with a neural module.
    """

    def __init__(self, initialized, semantic):
        self.initialized = initialized
        self.semantic = semantic

    def __compute_value__(self, dsl, environment):
        return self.semantic(**self.initialized, environment=environment)

    def all_state_values(self):
        return self.initialized.values()


def _create_module_for_type(module_factory, t):
    return lambda: module_factory(t)


def _inject_environment_argument(t):
    if isinstance(t, ArrowType):
        return lambda initialized_module, environment: lambda *args, **kwargs: initialized_module(
            *args, **kwargs, environment=environment
        )
    return lambda initialized_module, environment: initialized_module(
        environment=environment
    )


def create_modules(tag: str, types: List[Type], module_factory):
    """
    Create a dictionary of modules for a set of types, given the module factory.

    :param tag: Tag to use for the modules.
    :param types: Types to create modules for.
    :param module_factory: Function that creates a module given the input and output shapes.
    """
    return {t: (tag, _create_module_for_type(module_factory, t)) for t in types}


@internal_only
def compute_io_shape(t):
    """
    t : ArrowType
    returns: dict(input_shape, output_shape)
        input_shape: list of tuples (shape, type)
        output_shape: tuple (shape, type)
    """
    assert isinstance(t, ArrowType)
    input_types = t.input_type
    output_type = t.output_type

    def get_shape(t):
        match t:
            case TensorType(_, shape):
                return shape
            case ListType(element_type):
                return get_shape(element_type)
            case _:
                raise NotImplementedError(f"Cannot compute shape for type {t}")

    input_shape = [get_shape(t) for t in input_types]
    output_shape = get_shape(output_type)
    return input_shape, output_shape
