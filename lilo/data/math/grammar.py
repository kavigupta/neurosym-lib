"""
math: grammar.py | Author : Sagnik Anupam

Utility functions for loading Python DSLs for the math domain. This grammar was originally designed and used in the SuperUROP project "Neurosymbolic Reasoning for Math Domains" and can be found in the dreamcoder/math domain.
"""
from collections import OrderedDict

from src.models.model_loaders import ModelLoaderRegistries, GRAMMAR, ModelLoader
from src.models.laps_grammar import LAPSGrammar

import dreamcoder.domains.math.mathPrimitives as mathPrimitives

GrammarRegistry = ModelLoaderRegistries[GRAMMAR]
LARGEST_CONSTANT = 10 #Largest constant encoded in the math domain, must be between 0 and 25

@GrammarRegistry.register
class MathGrammarLoader(ModelLoader):
    """Loads the math grammar.
    Original source: dreamcoder/domains/math/mathPrimitives.
    """

    name = "math"  # Special handler for OCaml enumeration.

    def load_model(self, experiment_state):
        math_primitives = list(
            OrderedDict((x, True) for x in mathPrimitives.mathPrimitives(LARGEST_CONSTANT)).keys()
        )
        grammar = LAPSGrammar.uniform(math_primitives)
        return grammar