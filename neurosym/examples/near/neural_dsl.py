from dataclasses import dataclass
from typing import List

from neurosym.utils.documentation import internal_only

from ...dsl.dsl import DSL
from ...programs.hole import Hole
from ...programs.s_expression import InitializedSExpression, SExpression
from ...types.type import ArrowType, ListType, TensorType, Type
from .neural_hole_filler import DictionaryNeuralHoleFiller, NeuralHoleFiller


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

    neural_hole_filler: NeuralHoleFiller

    @classmethod
    def from_dsl(cls, dsl: DSL, neural_hole_filler: NeuralHoleFiller):
        """
        Creates a NeuralDSL from a DSL and a set of type specific modules.

        The type specific modules are used to fill holes in partial programs.

        :param dsl: The DSL to extend.
        :param neural_hole_filler: A ``NeuralHoleFiller`` object that
            maps types to neural modules.
        """

        return cls(
            productions=dsl.productions,
            valid_root_types=dsl.valid_root_types,
            max_type_depth=dsl.max_type_depth,
            max_env_depth=dsl.max_env_depth,
            neural_hole_filler=neural_hole_filler,
        )

    def initialize(self, program: SExpression) -> InitializedSExpression:
        """
        Initializes all the productions in the given program.

        Returns a new program with the same structure, but with all the productions
        initialized.
        """
        if isinstance(program, Hole):
            module = self.neural_hole_filler.initialize_module(program.twe)
            if module is None:
                raise PartialProgramNotFoundError(
                    f"Cannot initialize program {program}."
                )
            return _NeuralHole(
                program,
                {"initialized_module": module},
                _inject_environment_argument(program.twe.typ),
            )

        return super().initialize(program)


class _NeuralHole:
    """
    A hole that can be filled with a neural module.
    """

    def __init__(self, original_hole, initialized, semantic):
        self.original_hole = original_hole
        self.initialized = initialized
        self.semantic = semantic

    def __compute_value__(self, dsl, environment):
        del dsl
        return self.semantic(**self.initialized, environment=environment)

    def all_state_values(self):
        return self.initialized.values()

    def uninitialize(self) -> Hole:
        return self.original_hole


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


def create_modules(types: List[Type], module_factory):
    """
    Create a dictionary of modules for a set of types, given the module factory.

    :param types: Types to create modules for.
    :param module_factory: Function that creates a module given the input and output shapes.
    """
    return DictionaryNeuralHoleFiller(
        {t: _create_module_for_type(module_factory, t) for t in types}
    )


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
