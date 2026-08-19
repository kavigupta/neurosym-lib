"""
math: grammar.py | Author : Sagnik Anupam

Utility functions for loading Python DSLs for the math domain. This grammar was originally designed and used in the SuperUROP project "Neurosymbolic Reasoning for Math Domains" and can be found in the dreamcoder/math domain.
"""
from collections import OrderedDict

from src.models.model_loaders import ModelLoaderRegistries, GRAMMAR, ModelLoader
from src.models.laps_grammar import LAPSGrammar

import dreamcoder.domains.toy.toyPrimitives as toyPrimitives

GrammarRegistry = ModelLoaderRegistries[GRAMMAR]

@GrammarRegistry.register
class ToyGrammarLoader(ModelLoader):
    """Loads the math grammar.
    Original source: dreamcoder/domains/toy/toyPrimitives.
    """

    name = "toy"  # Special handler for OCaml enumeration.

    def load_model(self, experiment_state):
        toy_primitives = list(
            OrderedDict((x, True) for x in toyPrimitives.toyPrimitives()).keys()
        )
        grammar = LAPSGrammar.uniform(toy_primitives)
        return grammar