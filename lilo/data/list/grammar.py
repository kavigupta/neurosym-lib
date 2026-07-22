"""
list: grammar.py | Author : Sagnik Anupam

Utility functions for loading Python DSLs for the list domain.
"""
from collections import OrderedDict

from src.models.model_loaders import ModelLoaderRegistries, GRAMMAR, ModelLoader
from src.models.laps_grammar import LAPSGrammar

import dreamcoder.domains.list.listPrimitives as listPrimitives

GrammarRegistry = ModelLoaderRegistries[GRAMMAR]

@GrammarRegistry.register
class ListGrammarLoader(ModelLoader):
    """Loads the list grammar.
    Original source: dreamcoder/domains/list/lisPrimitives.
    """

    name = "list"  # Special handler for OCaml enumeration.

    def load_model(self, experiment_state):
        list_primitives = list(
            OrderedDict((x, True) for x in listPrimitives.primitives()).keys()
        )
        grammar = LAPSGrammar.uniform(list_primitives)
        return grammar