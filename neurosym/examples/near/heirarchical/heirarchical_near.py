from typing import Callable, Dict

from torch import nn

from neurosym.dsl.dsl import DSL
from neurosym.examples.near.cost import IdentityProgramEmbedding, NearCost
from neurosym.examples.near.heirarchical.repeated_refine import refinement_graph
from neurosym.examples.near.models.torch_program_module import TorchProgramModule
from neurosym.examples.near.neural_dsl import NeuralDSL
from neurosym.examples.near.neural_hole_filler import NeuralHoleFiller
from neurosym.examples.near.search_graph import validated_near_graph
from neurosym.types.type import Type
from neurosym.types.type_with_environment import Environment, TypeWithEnvironment


def heirarchical_near_graph(
    high_level_dsl: DSL,
    symbol: str,
    refined_dsl: DSL,
    typ: Type,
    validation_cost_creator: Callable[
        [Callable[[TorchProgramModule], nn.Module], Dict[str, float]], NearCost
    ],
    neural_hole_filler: NeuralHoleFiller,
    **near_params
):
    r"""
    Runs NEAR heirarchically. First finds a program using the high_level_dsl, then
    refines it using the refined_dsl, replacing any instances of symbol with a new
    search graph, using the `refined_dsl` as the language to replace the symbol with.

    :param high_level_dsl: The high level DSL to use for the initial search.
    :param symbol: The symbol to replace in the high level DSL.
    :param refined_dsl: The refined DSL to use for the replacement.
    :param typ: The type of the program to generate.
    :param validation_cost_creator: A function that creates a NearCost object.
    :param neural_hole_filler: The neural hole filler to use for the search.
    :param near_params: Additional parameters to pass to the search graph.

    :return: A ReturnSearchGraph object.

    Implements the following monadic pseudocode (Haskell)

    .. code-block:: haskell

        type Symbol = String

        hNear hlDSL cost symbol refinedDSL = do
            p_HL <- near hlDSL cost
            let ip_HL = freeze (initialize p_HL)
            p_concrete <- iterateUntilM' (not . hasSymbol symbol) ip_HL $ \ip -> do
                let node = withSymbol symbol ip
                let cost' fp = cost $ substitute ip node fp
                p_node <- near refinedDSL cost'
                let ip_node = initialize p_node
                return $ substitute ip node (freeze ip_node)
            return p_concrete

        (>>=) :: SearchGraph a -> (a -> SearchGraph b) -> SearchGraph b

    """
    overall_dsl = high_level_dsl.add_productions(*refined_dsl.productions)
    symbol_cost = refined_dsl.minimal_term_size_for_type(
        TypeWithEnvironment(
            high_level_dsl.get_production(symbol)
            .type_signature()
            .return_type_template(),
            Environment.empty(),
        )
    )

    g = validated_near_graph(
        NeuralDSL.from_dsl(dsl=high_level_dsl, neural_hole_filler=neural_hole_filler),
        typ,
        cost=validation_cost_creator(IdentityProgramEmbedding(), {symbol: symbol_cost}),
        **near_params,
    )
    g = g.bind(
        lambda res, cost: refinement_graph(
            refined_dsl,
            overall_dsl,
            res,
            cost,
            symbol,
            validation_cost_creator,
            neural_hole_filler,
            **near_params,
        )
    )
    return g
