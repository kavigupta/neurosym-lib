from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from neurosym.types.type_with_environment import TypeWithEnvironment

from ..programs.hole import Hole
from ..programs.s_expression import InitializedSExpression, SExpression
from ..types.type import Type

from .production import Production


@dataclass
class DSL:
    productions: List[Production]
    # variable_system: VariableSystem TODO(KG) add this

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

    def symbols(self):
        return self._production_by_symbol.keys()

    def arity(self, sym: str) -> int:
        """
        Returns the arity of the production with the given symbol.
        """
        return self.get_production(sym).type_signature().arity()

    def expansions_for_type(self, type: Type) -> List[SExpression]:
        """
        Possible expansions for the given type.

        An expansion is an SExpression with holes in it. The holes can be filled in with
        other SExpressions to produce a complete SExpression.
        """
        result = []
        for production in self.productions:
            arg_types = production.type_signature().unify_return(type)
            if arg_types is not None:
                result.append(
                    SExpression(
                        production.symbol(),
                        tuple(Hole.of(t) for t in arg_types),
                    )
                )
        return result

    def get_production(self, symbol: str) -> Production:
        """
        Return the production with the given symbol.
        """
        assert isinstance(symbol, str), f"Expected string, got {type(symbol)}: {symbol}"
        return self._production_by_symbol[symbol]

    def initialize(
        self, program: SExpression, hole_callback=None
    ) -> InitializedSExpression:
        """
        Initializes all the productions in the given program.

        Returns a new program with the same structure, but with all the productions
        initialized.
        """
        if isinstance(program, Hole):
            assert hole_callback is not None
            return hole_callback(program)
        if hasattr(program, "__initialize__"):
            return program.__initialize__(self)
        prod = self.get_production(program.symbol)
        return InitializedSExpression(
            program.symbol,
            tuple(self.initialize(child) for child in program.children),
            prod.initialize(self),
        )

    def compute_on_pytorch(self, program: InitializedSExpression):
        if hasattr(program, "__compute_value__"):
            return program.__compute_value__(self)
        prod = self.get_production(program.symbol)
        return prod.compute_on_pytorch(
            self,
            program.state,
            [self.compute_on_pytorch(child) for child in program.children],
        )

    def all_rules(self, *target_types) -> Dict[Type, List[Tuple[str, List[Type]]]]:
        """
        Returns a dictionary of all the rules in the DSL, where the keys are the types
        that can be expanded, and the values are a list of tuples of the form
        (symbol, types), where symbol is the symbol of the production, and types is a
        list of types that can be used to expand the given type.

        This is useful for generating a PCFG.
        """
        types_to_expand = list(target_types)
        rules = {}
        while len(types_to_expand) > 0:
            type = types_to_expand.pop()
            if type in rules:
                continue
            rules[type] = []
            for production in self.productions:
                types = production.type_signature().unify_return(type)
                if types is None:
                    continue
                rules[type].append((production.symbol(), types))
                types_to_expand.extend(types)
        return rules

    def validate_all_rules_reachable(self, *target_types):
        """
        Checks that all the rules in the DSL are reachable from at least one of the
        target types.
        """
        symbols = set()
        rules = self.all_rules(*target_types)
        for rule in rules.values():
            symbols.update([symbol for symbol, _ in rule])
        for production in self.productions:
            assert production.symbol() in symbols, (
                f"Production {production.symbol()} is unreachable from target types "
                f"{target_types}"
            )

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
