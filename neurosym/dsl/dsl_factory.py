from typing import Dict, List
from neurosym.dsl.dsl import DSL
from neurosym.dsl.production import (
    ConcreteProduction,
    ParameterizedProduction,
    Production,
)
from neurosym.types.type_string_repr import TypeDefiner, parse_type
import numpy as np
from neurosym.types.type_signature import (
    ConcreteTypeSignature,
    expansions,
    type_universe,
)
from neurosym.types.type import ArrowType, ListType
from neurosym.types.type_string_repr import render_type


class DSLFactory:
    """
    A factory for creating DSLs.

    Example usage:
    ```
    factory = DSLFactory()
    factory.typedef("fn", "(i) -> i")
    factory.concrete("inc", "$fn", lambda x: x + 1)
    factory.concrete("const_0", "$fn", lambda x: 0)
    factory.concrete("compose", "($fn, $fn) -> $fn", lambda f, g: lambda x: f(g(x))
    factory.finalize()
    """

    def __init__(self, max_expansion_steps=np.inf, max_overall_depth=6, **env):
        self.t = TypeDefiner(**env)
        self._concrete_productions = []
        self._parameterized_productions = []
        self._signatures = []
        self.max_expansion_steps = max_expansion_steps
        self.max_overall_depth = max_overall_depth
        self.prune = False
        self.target_types = None

    def typedef(self, key, type_str):
        """
        Define a type.
        """
        self.t.typedef(key, type_str)

    def concrete(self, symbol, type_str, fn):
        """
        Add a concrete production to the DSL.
        """
        sig = self.t.sig(type_str)
        self._concrete_productions.append(
            (
                symbol,
                sig,
                fn,
            )
        )
        self._signatures.append(sig)

    def parameterized(self, symbol, type_str, fn, parameters):
        """
        Add a parameterized production to the DSL.
        """
        sig = self.t.sig(type_str)
        self._parameterized_productions.append(
            (
                symbol,
                sig,
                fn,
                parameters,
            )
        )
        self._signatures.append(sig)

    def prune_to(self, *target_types):
        """
        Prune the DSL to only include productions that can be constructed from the given
        target types.
        """
        self.prune = True
        self.target_types = [self.t(x) for x in target_types]

    def _expansions_for_single_production(
        self, terminals, type_constructors, constructor, symbol, sig, *args
    ):
        sigs = list(
            expansions(
                sig.astype(),
                terminals,
                type_constructors,
                max_expansion_steps=self.max_expansion_steps,
                max_overall_depth=self.max_overall_depth,
            )
        )
        assert len(sigs) > 0, f"No expansions within depth/step bounds for {symbol}"

        prods = [
            constructor(symbol, ConcreteTypeSignature.from_type(expansion), *args)
            for expansion in sigs
        ]

        return {symbol: Production.reindex(prods)}

    def _expansions_for_all_productions(
        self, expand_to, terminals, type_constructors, args
    ):
        result = {}
        for arg in args:
            result.update(
                self._expansions_for_single_production(
                    expand_to, terminals, type_constructors, *arg
                )
            )
        return result

    def finalize(self):
        """
        Finalize the DSL.
        """

        known_types = [x.astype() for x in self._signatures]

        universe = type_universe(known_types)

        sym_to_productions: Dict[str, List[Production]] = {}
        sym_to_productions.update(
            self._expansions_for_all_productions(
                *universe, ConcreteProduction, self._concrete_productions
            )
        )
        sym_to_productions.update(
            self._expansions_for_all_productions(
                *universe, ParameterizedProduction, self._parameterized_productions
            )
        )

        make_dsl = lambda: DSL(
            [prod for prods in sym_to_productions.values() for prod in prods]
        )
        dsl = make_dsl()
        if self.prune:
            assert self.target_types is not None
            symbols = dsl.constructible_symbols(*self.target_types)
            new_sym_to_productions = {}
            for original_symbol, prods in sym_to_productions.items():
                new_sym_to_productions[original_symbol] = [
                    x for x in prods if x.symbol() in symbols
                ]
                if len(new_sym_to_productions[original_symbol]) == 0:
                    raise TypeError(
                        f"All productions for {original_symbol} were pruned. "
                        f"Check that the target types are correct."
                    )
                new_sym_to_productions[original_symbol] = Production.reindex(
                    new_sym_to_productions[original_symbol]
                )
            sym_to_productions = new_sym_to_productions
            dsl = make_dsl()
        return dsl
