from dataclasses import dataclass
from types import NoneType
from typing import Callable, Dict, Iterator, List, Tuple, Union

import numpy as np

from neurosym.types.type_annotated_object import TypeAnnotatedObject
from neurosym.types.type_with_environment import (
    Environment,
    PermissiveEnvironmment,
    TypeWithEnvironment,
)
from neurosym.utils.tree_trie import TreeTrie

from ..programs.hole import Hole
from ..programs.s_expression import InitializedSExpression, SExpression
from ..types.type import GenericTypeVariable, Type
from .production import Production

ROOT_SYMBOL = "<root>"


@dataclass
class DSL:
    """
    Represents a domain-specific language (DSL) for generating programs.
    """

    # list of productions
    productions: List[Production]
    # list of types that can be used as the root of a program. If None, all types can be
    # used as the root.
    valid_root_types: Union[List[Type], NoneType]
    # max depth of a valid type for a program in this DSL
    max_type_depth: float
    # max depth of a valid environment for a program in this DSL
    max_env_depth: float

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
                is_wildcard_predicate=lambda x: isinstance(x, GenericTypeVariable),
            )

    def symbols(self):
        """
        Returns a set of all the symbols in the DSL.
        """
        return self._production_by_symbol.keys()

    def ordered_symbols(self, *, include_root=False):
        """
        Returns a list of all the symbols in the DSL, ordered by symbol.

        :param include_root: Whether to include the root symbol in the list.
        """
        symbols = sorted(self.symbols())
        if include_root:
            symbols = [ROOT_SYMBOL] + symbols
        return symbols

    def arity(self, sym: str) -> int:
        """
        Returns the arity of the production with the given symbol.
        """
        return self.get_production(sym).type_signature().arity()

    def productions_for_type(
        self, typ: TypeWithEnvironment
    ) -> Iterator[Tuple[Production, List[TypeWithEnvironment]]]:
        """
        Compute all the productions that can be applied to the given type.

        Returns a generator of tuples of the form ``(production, arg_types)``, where
        ``production`` is a production that can be applied to the given type, and ``arg_types``
        is a list of types to fill in the holes of the production.
        """
        for idx in sorted(self._out_type_to_prod_idx.query(typ.typ)):
            production = self.productions[idx]
            arg_types = production.type_signature().unify_return(typ)
            if arg_types is not None:
                yield production, arg_types

    def expansions_for_type(self, typ: TypeWithEnvironment) -> List[SExpression]:
        """
        Possible expansions for the given type.

        An expansion is an :py:class:`ns.SExpression` with holes in it. The holes can be filled in with
        other ``SExpressions`` to produce a complete ``SExpression``.

        :param typ: The type to expand.
        """
        return [
            SExpression(
                production.symbol(),
                tuple(Hole.of(t) for t in arg_types),
            )
            for production, arg_types in self.productions_for_type(typ)
        ]

    def get_production(self, symbol: str) -> Production:
        """
        Return the production with the given symbol.

        :param symbol: The symbol of the production to return.
        """
        assert isinstance(symbol, str), f"Expected string, got {type(symbol)}: {symbol}"
        return self._production_by_symbol[symbol]

    def initialize(self, program: SExpression) -> InitializedSExpression:
        """
        Initializes all the productions in the given program.

        Returns a new program with the same structure, but with all the productions
        initialized.

        :param program: The program to initialize.
        """
        if hasattr(program, "__initialize__"):
            return program.__initialize__(self)
        prod = self.get_production(program.symbol)
        return InitializedSExpression(
            program.symbol,
            tuple(self.initialize(child) for child in program.children),
            prod.initialize(self),
        )

    def compute(
        self,
        program: InitializedSExpression,
        environment: Tuple[TypeAnnotatedObject] = (),
    ):
        """
        Compute the value of the given program.
        """
        if hasattr(program, "__compute_value__"):
            return program.__compute_value__(self)
        prod = self.get_production(program.symbol)
        return prod.apply(
            self, program.state, program.children, environment=environment
        )

    def all_rules(
        self, care_about_variables, valid_root_types=None
    ) -> Dict[Type, List[Tuple[str, List[Type]]]]:
        """
        Returns a dictionary of all the rules in the DSL, where the keys are the types
        that can be expanded, and the values are a list of tuples of the form
        (symbol, types), where symbol is the symbol of the production, and types is a
        list of types that can be used to expand the given type.

        This is useful for generating a PCFG.

        :param care_about_variables: Whether to care about variables when enumerating rules.
        :param valid_root_types: The types that can be used as the root of a program. If None,
            we use the valid_root_types of the DSL.

        :return: A dictionary of rules. If care_about_variables is False, the keys are
            the types that can be expanded, and the values are a list of tuples of the form
            (symbol, types), where symbol is the symbol of the production, and types is a
            list of types that can be used to expand the given type. If care_about_variables
            is True, the keys are TypeWithEnvironment objects, and the values are lists of
            tuples of the form (symbol, TypeWithEnvironment), where symbol is the symbol of
            the production, and TypeWithEnvironment is the type of the production.
        """
        if valid_root_types is None:
            if self.valid_root_types is not None:
                valid_root_types = self.valid_root_types
            else:
                raise ValueError(
                    "Cannot generate rules for a DSL with no valid root types."
                )
        twes_to_expand = [
            TypeWithEnvironment(
                type,
                (
                    Environment.empty()
                    if care_about_variables
                    else PermissiveEnvironmment()
                ),
            )
            for type in valid_root_types
        ]
        rules = {}
        while len(twes_to_expand) > 0:
            twe = twes_to_expand.pop()
            if (
                twe.typ.depth > self.max_type_depth
                or len(twe.env) > self.max_env_depth
                or twe in rules
            ):
                continue
            rules[twe] = []
            for prod, twes in self.productions_for_type(twe):
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

    def constructible_symbols(self, care_about_variables, valid_root_types=None):
        """
        Returns all the symbols that can be constructed from the given target types.

        :param care_about_variables: Whether to care about variables when computing constructible symbols.
        :param valid_root_types: The types that can be used as the root of a program. If None,
            we use the valid_root_types of the DSL.

        :return: A set of symbols that can be constructed from the given target types.
        """
        type_to_rules = self.all_rules(
            care_about_variables=care_about_variables,
            valid_root_types=valid_root_types,
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

        :param program: The program to compute the type of.
        :param lookup: A function that can be used to look up the type of a program. This
            is useful for handling special caes. If the function returns None, we compute
            the type of the program using the DSL.
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
        """
        Add a production to the DSL.
        """
        return DSL(
            self.productions + [prod],
            self.valid_root_types,
            self.max_type_depth,
            self.max_env_depth,
        )

    def with_valid_root_types(self, valid_root_types):
        """
        Return a copy of this DSL with the given valid root types.
        """
        return DSL(
            self.productions,
            valid_root_types,
            self.max_type_depth,
            self.max_env_depth,
        )

    def create_smoothing_mask(self, dsl_subset):
        """
        Create a mask that can be used to smooth a distribution over the full DSL
        without adding weight to symbols that are not in the subset.
        """
        symbols_full = self.ordered_symbols(include_root=True)
        symbols_subset = set(dsl_subset.ordered_symbols(include_root=True))
        return np.array([s in symbols_subset for s in symbols_full])
