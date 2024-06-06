from dataclasses import dataclass
from typing import Callable, Dict, Tuple

from torch import nn

from neurosym.types.type_signature import FunctionTypeSignature

from ...dsl.dsl import DSL
from ...dsl.production import ParameterizedProduction, Production
from ...programs.hole import Hole
from ...programs.s_expression import InitializedSExpression, SExpression
from ...types.type import ArrowType, ListType, TensorType, Type


class PartialProgramNotFoundError(Exception):
    """
    Raised when a partial program cannot be found.
    """


@dataclass
class NeuralDSL(DSL):
    """
    A neural DSL extends `DSL` to handle neural heuristics (ie: type-appropriate NN productions)
    These neural heuristics can be used to fill holes in partial programs.
    Required to run NEAR.
    """

    # partial_programs: Dict[Type, SExpression]
    type_to_symbol: Dict[Type, str]

    @classmethod
    def from_dsl(
        cls, dsl: DSL, modules: Dict[Type, Tuple[str, Callable[[], nn.Module]]]
    ):
        """
        Creates a NeuralDSL from a DSL and a set of type specific modules.

        The type specific modules are used to fill holes in partial programs.

        Args:
            dsl: The DSL to extend.
            modules: A dictionary mapping types to tags and functions that
                are used to initialize the modules for that type.

        Returns:
            A NeuralDSL.
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
                lambda initialized_module: initialized_module,
                index=None,
                initializers=dict(initialized_module=module_template),
            )

            partial_productions.append(module_c_prod)

        productions = dsl.productions + partial_productions

        return cls(
            productions=productions,
            valid_root_types=dsl.valid_root_types,
            max_type_depth=dsl.max_type_depth,
            max_env_depth=dsl.max_env_depth,
            type_to_symbol=type_to_symbol,
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


def create_module_for_type(module_factory, t):
    shape = compute_io_shape(t)
    return lambda: module_factory(*shape)


def create_modules(tag, types, module_factory):
    return {t: (tag, create_module_for_type(module_factory, t)) for t in types}


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
