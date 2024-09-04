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
        partial_productions = []
        type_to_symbol = {}

        count_by_tag = {}
        for fn_type, (tag, module_template) in modules.items():
            assert isinstance(
                fn_type, ArrowType
            ), f"Type of partial NN module must be an ArrowType, got {fn_type}"
            count_by_tag[tag] = count_by_tag.get(tag, 0) + 1
            identifier = f"__neural_dsl_internal_{tag}_{count_by_tag[tag]}"
            type_to_symbol[fn_type] = identifier
            # pylint: disable=unexpected-keyword-arg
            module_c_prod = ParameterizedProduction(
                identifier,
                FunctionTypeSignature([], fn_type),
                lambda initialized_module, environment: lambda *args, **kwargs: initialized_module(
                    *args, **kwargs, environment=environment
                ),
                index=None,
                initializers=dict(initialized_module=module_template),
                provide_enviroment="environment",
            )

            partial_productions.append(module_c_prod)

        productions = dsl.productions + partial_productions

        return cls(
            productions=productions,
            valid_root_types=dsl.valid_root_types,
            max_type_depth=dsl.max_type_depth,
            max_env_depth=dsl.max_env_depth,
            type_to_symbol=type_to_symbol,
            original_symbols=set(dsl.symbols()),
        )

    def get_partial_program(self, hole: Hole) -> Production:
        """
        Returns a production that can be used to fill the given hole.
        """
        return SExpression(
            self.type_to_symbol[hole.twe.typ],
            [],
        )

    def initialize(self, program: SExpression) -> InitializedSExpression:
        """
        Initializes all the productions in the given program.

        Returns a new program with the same structure, but with all the productions
        initialized.
        """
        if isinstance(program, Hole):
            try:
                prog = self.get_partial_program(program)
            except KeyError as e:
                raise PartialProgramNotFoundError(
                    f"Cannot initialize program {program}."
                ) from e
        else:
            prog = program

        if hasattr(prog, "__initialize__"):
            return prog.__initialize__(self)
        prod = self.get_production(prog.symbol)
        return InitializedSExpression(
            prog.symbol,
            tuple(self.initialize(child) for child in prog.children),
            prod.initialize(self),
        )

    def program_has_no_holes(self, program: Union[SExpression, DSLSearchNode]) -> bool:
        """
        Returns True if the given program has no holes.
        """
        if isinstance(program, DSLSearchNode):
            program = program.program
        assert isinstance(program, SExpression)
        return symbols_for_program(program) - self.original_symbols == set()


def _create_module_for_type(module_factory, t):
    return lambda: module_factory(t)


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
