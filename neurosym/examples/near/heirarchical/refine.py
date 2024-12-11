from neurosym.dsl.dsl import DSL
from neurosym.examples.near.models.torch_program_module import TorchProgramModule
from neurosym.programs.s_expression import InitializedSExpression

from .utils import replace_first


class RefinementEmbedding:
    """
    Represents a refinement embedding that places the given program in some main
    program, replacing a specified symbol.
    """

    def __init__(
        self,
        symbol_to_replace: str,
        main_program: InitializedSExpression,
        overall_dsl: DSL,
    ):
        self.to_replace = symbol_to_replace
        self.frozen = main_program
        self.overall_dsl = overall_dsl

    def __call__(self, program_module):
        frozen_subst, replaced = replace_first(
            self.frozen, self.to_replace, program_module.initalized_program
        )
        assert replaced
        return TorchProgramModule(self.overall_dsl, frozen_subst)
