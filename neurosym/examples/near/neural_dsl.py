from dataclasses import dataclass
from typing import Callable, Dict, List, Union

from torch import nn

from neurosym.search_graph.dsl_search_node import DSLSearchNode
from neurosym.utils.documentation import internal_only

from ...dsl.dsl import DSL
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

    modules: Dict[Type, Callable[[], nn.Module]]

    @classmethod
    def from_dsl(cls, dsl: DSL, modules: Dict[Type, Callable[[], nn.Module]]):
        """
        Creates a NeuralDSL from a DSL and a set of type specific modules.

        The type specific modules are used to fill holes in partial programs.

        :param dsl: The DSL to extend.
        :param modules: A dictionary mapping types to tags and functions that
            are used to initialize the modules for that type.
        """

        return cls(
            productions=dsl.productions,
            valid_root_types=dsl.valid_root_types,
            max_type_depth=dsl.max_type_depth,
            max_env_depth=dsl.max_env_depth,
            modules=modules,
        )

    def initialize(self, program: SExpression) -> InitializedSExpression:
        """
        Initializes all the productions in the given program.

        Returns a new program with the same structure, but with all the productions
        initialized.
        """
        if isinstance(program, Hole):
            if program.twe.typ not in self.modules:
                raise PartialProgramNotFoundError(
                    f"Cannot initialize program {program}."
                )
            module_template = self.modules[program.twe.typ]
            initialized = {"initialized_module": module_template()}
            return _NeuralHole(
                initialized, _inject_environment_argument(program.twe.typ)
            )

        return super().initialize(program)


class _NeuralHole:
    """
    A hole that can be filled with a neural module.
    """

    def __init__(self, initialized, semantic):
        self.initialized = initialized
        self.semantic = semantic

    def __compute_value__(self, dsl, environment):
        del dsl
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


def create_modules(types: List[Type], module_factory):
    """
    Create a dictionary of modules for a set of types, given the module factory.

    :param types: Types to create modules for.
    :param module_factory: Function that creates a module given the input and output shapes.
    """
    return {t: _create_module_for_type(module_factory, t) for t in types}


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
