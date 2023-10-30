from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from neurosym.types.type_with_environment import (
    Environment,
    PermissiveEnvironmment,
    TypeWithEnvironment,
)
from neurosym.utils.tree_trie import TreeTrie

from ..programs.hole import Hole
from ..programs.s_expression import InitializedSExpression, SExpression
from ..types.type import Type, TypeVariable
from .production import Production


@dataclass
class DSL:
    productions: List[Production]

    def __post_init__(self):
        symbols = set()
        for production in self.productions:
            assert (
                production.symbol() not in symbols
            ), f"Duplicate symbol {production.symbol()}"
            symbols.add(production.symbol())
        self._production_by_symbol = {
            production.symbol(): production for production in self.productions
        }

        self._out_type_to_prod_idx = TreeTrie.empty()
        for i, prod in enumerate(self.productions):
            self._out_type_to_prod_idx.insert(
                prod.type_signature().return_type_template(),
                i,
                is_wildcard_predicate=lambda x: isinstance(x, TypeVariable),
            )

    def symbols(self):
        return self._production_by_symbol.keys()

    def arity(self, sym: str) -> int:
        """
        Returns the arity of the production with the given symbol.
        """
        return self.get_production(sym).type_signature().arity()

    def _productions_for_type(
        self, typ: TypeWithEnvironment
    ) -> List[Tuple[Production, List[TypeWithEnvironment]]]:
        for idx in sorted(self._out_type_to_prod_idx.query(typ.typ)):
            production = self.productions[idx]
            arg_types = production.type_signature().unify_return(typ)
            if arg_types is not None:
                yield production, arg_types

    def expansions_for_type(self, typ: TypeWithEnvironment) -> List[SExpression]:
        """
        Possible expansions for the given type.

        An expansion is an SExpression with holes in it. The holes can be filled in with
        other SExpressions to produce a complete SExpression.
        """
        return [
            SExpression(
                production.symbol(),
                tuple(Hole.of(t) for t in arg_types),
            )
            for production, arg_types in self._productions_for_type(typ)
        ]

    def get_production(self, symbol: str) -> Production:
        """
        Return the production with the given symbol.
        """
        assert isinstance(symbol, str), f"Expected string, got {type(symbol)}: {symbol}"
        return self._production_by_symbol[symbol]

    def initialize(self, program: SExpression) -> InitializedSExpression:
        """
        Initializes all the productions in the given program.

        Returns a new program with the same structure, but with all the productions
        initialized.
        """
        if hasattr(program, "__initialize__"):
            return program.__initialize__(self)
        prod = self.get_production(program.symbol)
        return InitializedSExpression(
            program.symbol,
            tuple(self.initialize(child) for child in program.children),
            prod.initialize(self),
        )

    def compute(self, program: InitializedSExpression):
        if hasattr(program, "__compute_value__"):
            return program.__compute_value__(self)
        prod = self.get_production(program.symbol)
        return prod.apply(self, program.state, program.children)

    def all_rules(
        self, *target_types: Tuple[Type], care_about_variables, type_depth_limit
    ) -> Dict[Type, List[Tuple[str, List[Type]]]]:
        """
        Returns a dictionary of all the rules in the DSL, where the keys are the types
        that can be expanded, and the values are a list of tuples of the form
        (symbol, types), where symbol is the symbol of the production, and types is a
        list of types that can be used to expand the given type.

        This is useful for generating a PCFG.
        """
        twes_to_expand = [
            TypeWithEnvironment(
                type,
                Environment.empty()
                if care_about_variables
                else PermissiveEnvironmment(),
            )
            for type in target_types
        ]
        rules = {}
        while len(twes_to_expand) > 0:
            twe = twes_to_expand.pop()
            if twe.typ.depth > type_depth_limit or twe in rules:
                continue
            rules[twe] = []
            for prod, twes in self._productions_for_type(twe):
                rules[twe].append((prod.symbol(), twes))
                twes_to_expand.extend(twes)
        if not care_about_variables:
            rules = {
                out_twe.typ: [
                    (sym, [inp_twe.typ for inp_twe in inp_twes])
                    for sym, inp_twes in rules
                ]
                for out_twe, rules in rules.items()
            }
        return rules

    def constructible_symbols(
        self, *target_types, care_about_variables, type_depth_limit
    ):
        """
        Returns all the symbols that can be constructed from the given target types.
        """
        type_to_rules = self.all_rules(
            *target_types,
            care_about_variables=care_about_variables,
            type_depth_limit=type_depth_limit,
        )

        constructible = set()

        while True:
            done = True
            for out_t, rules in type_to_rules.items():
                if out_t in constructible:
                    continue
                for _, in_t in rules:
                    if set(in_t).issubset(constructible):
                        constructible.add(out_t)
                        done = False
            if done:
                break
        return {
            sym
            for _, rules in type_to_rules.items()
            for sym, in_t in rules
            if all(t in constructible for t in in_t)
        }

    def compute_type(
        self,
        program: SExpression,
        lookup: Callable[[SExpression], TypeWithEnvironment] = lambda x: None,
    ) -> TypeWithEnvironment:
        """
        Computes the type of the given program.
        """
        if lookup is not None:
            res = lookup(program)
            if res is not None:
                return res

        child_types = [self.compute_type(child, lookup) for child in program.children]
        prod = self.get_production(program.symbol)
        return prod.type_signature().unify_arguments(child_types)

    def render(self) -> str:
        """
        Render this DSL as a string.
        """
        return "\n".join(production.render() for production in self.productions)

    def add_production(self, prod):
        return DSL(self.productions + [prod])
