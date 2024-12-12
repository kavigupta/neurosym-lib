from typing import Callable

import torch.nn as nn

from neurosym.dsl.dsl import DSL
from neurosym.examples.near.models.torch_program_module import TorchProgramModule
from neurosym.examples.near.neural_dsl import NeuralDSL
from neurosym.examples.near.neural_hole_filler import NeuralHoleFiller
from neurosym.examples.near.search_graph import validated_near_graph
from neurosym.examples.near.validation import ValidationCost
from neurosym.programs.s_expression import InitializedSExpression
from neurosym.programs.s_expression_render import render_s_expression
from neurosym.search_graph.return_search_graph import ReturnSearchGraph


def refinement_graph(
    sub_dsl: DSL,
    overall_dsl: DSL,
    current_program: InitializedSExpression,
    cost,
    symbol_to_replace: str,
    validation_cost_creator: Callable[
        [Callable[[TorchProgramModule], nn.Module]], ValidationCost
    ],
    neural_hole_filler: NeuralHoleFiller,
    **near_params,
):
    u = current_program.uninitialize()
    if symbol_to_replace not in {x.symbol for x in u.postorder}:
        return ReturnSearchGraph(current_program, cost)

    g = validated_near_graph(
        NeuralDSL.from_dsl(dsl=sub_dsl, neural_hole_filler=neural_hole_filler),
        overall_dsl.get_production(symbol_to_replace)
        .type_signature()
        .astype()
        .output_type,
        cost=validation_cost_creator(
            _RefinementEmbedding(symbol_to_replace, current_program, overall_dsl)
        ),
        **near_params,
    )

    def after_search(result, cost_result):
        result = result.initalized_program
        _freeze(result)
        replaced, worked = current_program.replace_first(symbol_to_replace, result)
        assert worked
        print(
            "Refined",
            render_s_expression(current_program.uninitialize()),
            "at",
            symbol_to_replace,
        )
        print("to", render_s_expression(result.uninitialize()))

        return refinement_graph(
            sub_dsl,
            overall_dsl,
            replaced,
            cost_result,
            symbol_to_replace,
            validation_cost_creator,
            neural_hole_filler,
            **near_params,
        )

    return g.bind(after_search)


def _freeze(program):
    for state in program.all_state_values():
        for p in state.parameters():
            p.requires_grad = False


class _RefinementEmbedding:
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
        frozen_subst, replaced = self.frozen.replace_first(
            self.to_replace, program_module.initalized_program
        )
        assert replaced
        return TorchProgramModule(self.overall_dsl, frozen_subst)
