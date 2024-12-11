from typing import Callable

import torch.nn as nn

from neurosym.dsl.dsl import DSL
from neurosym.examples.near.heirarchical.repeated_refine import refinement_graph
from neurosym.examples.near.models.torch_program_module import TorchProgramModule
from neurosym.examples.near.search_graph import validated_near_graph
from neurosym.examples.near.validation import ValidationCost
from neurosym.types.type import Type
from tests.near.test_piecewise_linear import get_neural_dsl


def heirarchical_near_graph(
    high_level_dsl: DSL,
    symbol: str,
    refined_dsl: DSL,
    typ: Type,
    validation_cost_creator: Callable[
        [DSL, Callable[[TorchProgramModule], nn.Module]], ValidationCost
    ],
    **near_params
):
    overall_dsl = high_level_dsl.add_productions(*refined_dsl.productions)
    g = validated_near_graph(
        get_neural_dsl(high_level_dsl),
        typ,
        cost=validation_cost_creator(high_level_dsl, lambda x: x),
        **near_params,
    )
    g = g.bind(
        lambda res, cost: refinement_graph(
            refined_dsl,
            overall_dsl,
            res.initalized_program,
            cost,
            symbol,
            lambda func: validation_cost_creator(refined_dsl, func),
            **near_params,
        )
    )
    return g
