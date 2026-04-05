import copy
import warnings
from typing import Callable, Dict, List, Tuple

from ..types.type import Type
from ..types.type_signature import LambdaTypeSignature, VariableTypeSignature
from ..types.type_string_repr import TypeDefiner
from .dsl import DSL
from .production import (
    LambdaProduction,
    ParameterizedProduction,
    Production,
    VariableProduction,
)


class DSLFactory:
    """
    A factory for creating DSLs.

    Example usage:

    .. highlight:: python
    .. code-block:: python

        dslf = DSLFactory()
        dslf.typedef("fn", "(i) -> i")
        dslf.production("inc", "$fn", lambda x: x + 1)
        dslf.production("const_0", "$fn", lambda x: 0)
        dslf.production("compose", "($fn, $fn) -> $fn", lambda f, g: lambda x: f(g(x))
        dslf.finalize()
    """

    def __init__(
        self, max_env_depth=4, max_overall_depth=6, max_expansion_steps=None, **env
    ):
        del max_expansion_steps
        self.t = TypeDefiner(**env)
        self._parameterized_productions = []
        self._has_lambdas = False
        self.max_overall_depth = max_overall_depth
        self.max_env_depth = max_env_depth
        self.prune = False
        self.target_types = None
        self.prune_variables = False
        self.tolerate_pruning_entire_productions = False
        self._extra_productions = []

    def typedef(self, key: str, type_str: str):
        """
        Define a type with the given type string.
        The key will be used to refer to the type in future calls
        with a $ prefix. E.g.,

        .. highlight:: python
        .. code-block:: python

            dslf.typedef("fn", "(i) -> i")
            dslf.production("inc", "$fn", lambda x: x + 1)
        """
        self.t.typedef(key, type_str)

    def filtered_type_variable(self, key, type_filter: Callable[[Type], bool]):
        """
        Define a filtered type variable. This is a type variable that can only be
        instantiated with types that satisfy the given filter. The key will be used to
        refer to the type in future calls with a % prefix. E.g.,

        .. highlight:: python
        .. code-block:: python

            dslf.filtered_type_variable(
                "num", lambda x: isinstance(x, ns.AtomicType) and x.name in ["i", "f"]
            )
            dslf.production("+", "%num -> %num -> %num", lambda x: x)
        """
        self.t.filtered_type_variable(key, type_filter)

    def known_types(self, *types: Tuple[str, ...]):
        """
        No longer needed. Type variables are kept unexpanded, so the
        type universe does not need to be specified.
        """
        del types

    def no_zeroadic(self):
        """
        No longer needed. Type variables are kept unexpanded, so
        zeroadic type filtering is not applicable.
        """

    def lambdas(self, **kwargs):
        """
        Add lambda and variable productions to the DSL. Creates one lambda
        production per possible function arity (up to ``max_env_depth``),
        and one variable production per de Bruijn index.
        """
        del kwargs
        self._has_lambdas = True

    def extra_productions(
        self, symbol: str, productions: List[Production], stable: bool = True
    ):
        """
        Add custom productions to the DSL. These are added as-is without
        type expansion. If stable is True, these productions will not be
        reindexed during pruning.

        :param symbol: The symbol group name for these productions (e.g., "<shield>").
        :param productions: The list of productions to add.
        :param stable: If True, these productions will not be reindexed during pruning.
        """
        self._extra_productions.append((symbol, productions, stable))

    def concrete(self, symbol: str, type_str: str, semantics: object):
        """
        Deprecated alias of :py:meth:`production`.
        """
        warnings.warn(
            "The method concrete is deprecated. Use production instead.",
            DeprecationWarning,
        )
        self.production(symbol, type_str, semantics, {})

    def parameterized(
        self,
        symbol: str,
        type_str: str,
        semantics: object,
        parameters: Dict[str, Callable[[], object]],
    ):
        """
        Deprecated alias of :py:meth:`production`.
        """
        warnings.warn(
            "The method parameterized is deprecated. Use production instead.",
            DeprecationWarning,
        )
        self.production(symbol, type_str, semantics, parameters)

    def production(
        self,
        symbol: str,
        type_str: str,
        semantics: object,
        parameters: Dict[str, Callable[[], object]] = None,
    ):
        """
        Add a parameterized production to the DSL.

        :param symbol: The symbol for the production.
        :param type_str: The type string for the production.
        :param semantics: The semantics to use for the production. This should have
            a type corresponding to ``type_str``. Note: *this is not checked*.
        :param parameters: A dictionary mapping parameter names to functions that
            generate initial parameter values.
        """
        if parameters is None:
            parameters = {}
        sig = self.t.sig(type_str)
        self._parameterized_productions.append(
            (
                symbol,
                sig,
                semantics,
                parameters,
            )
        )

    def prune_to(
        self,
        *target_types: Tuple[str, ...],
        prune_variables=True,
        tolerate_pruning_entire_productions=False,
    ):
        """
        Direct the current DSLFactory to prune any productions p such that there does not exist some
        program s and type t in target_types such that s :: t and s contains p as a production.
        """
        self.prune = True
        self.target_types = [self.t(x) for x in target_types]
        self.prune_variables = prune_variables
        self.tolerate_pruning_entire_productions = tolerate_pruning_entire_productions

    @staticmethod
    def _create_productions_without_expansion(production_constructor, args):
        """
        Create one production per declaration, keeping type variables unexpanded.
        """
        result = {}
        for symbol, sig, *rest in args:
            prod = production_constructor(symbol, sig, *rest)
            if symbol in result:
                if result[symbol] != [prod]:
                    raise ValueError(f"Duplicate declarations for production: {symbol}")
            else:
                result[symbol] = [prod]
        return result

    def finalize(self) -> DSL:
        """
        Produce the DSL from this factory. This will generate all productions and
        potentially raise errors if there were issues with the way the DSL was
        constructed.
        """

        sym_to_productions: Dict[str, List[Production]] = {}
        sym_to_productions.update(
            self._create_productions_without_expansion(
                ParameterizedProduction.of, self._parameterized_productions
            )
        )

        stable_symbols = set()

        if self._has_lambdas:
            # One lambda production per possible function arity (0 to max_env_depth)
            sym_to_productions["<lambda>"] = [
                LambdaProduction(num_args, LambdaTypeSignature(num_args))
                for num_args in range(self.max_env_depth + 1)
            ]

            # One variable production per de Bruijn index
            sym_to_productions["<variable>"] = [
                VariableProduction(VariableTypeSignature(index_in_env))
                for index_in_env in range(self.max_env_depth)
            ]
            stable_symbols.add("<variable>")

        for symbol, prods, stable in self._extra_productions:
            sym_to_productions[symbol] = prods
            if stable:
                stable_symbols.add(symbol)

        if self.prune:
            assert self.target_types is not None
            sym_to_productions = _prune(
                sym_to_productions,
                self.target_types,
                care_about_variables=False,
                type_depth_limit=self.max_overall_depth,
                env_depth_limit=self.max_env_depth,
                stable_symbols=stable_symbols,
                tolerate_pruning_entire_productions=self.tolerate_pruning_entire_productions,
            )
            if self.prune_variables:
                sym_to_productions = _prune(
                    sym_to_productions,
                    self.target_types,
                    care_about_variables=True,
                    type_depth_limit=self.max_overall_depth,
                    env_depth_limit=self.max_env_depth,
                    stable_symbols=stable_symbols,
                    tolerate_pruning_entire_productions=self.tolerate_pruning_entire_productions,
                )
        dsl = _make_dsl(
            sym_to_productions,
            copy.copy(self.target_types),
            self.max_overall_depth,
            self.max_env_depth,
        )
        return dsl


def _make_dsl(sym_to_productions, valid_root_types, max_type_depth, max_env_depth):
    return DSL(
        [prod for prods in sym_to_productions.values() for prod in prods],
        valid_root_types,
        max_type_depth,
        max_env_depth=max_env_depth,
    )


def _prune(
    sym_to_productions,
    target_types,
    *,
    care_about_variables,
    type_depth_limit,
    env_depth_limit,
    stable_symbols,
    tolerate_pruning_entire_productions,
):
    dsl = _make_dsl(sym_to_productions, target_types, type_depth_limit, env_depth_limit)
    symbols = dsl.constructible_symbols(care_about_variables=care_about_variables)
    new_sym_to_productions = {}
    for original_symbol, prods in sym_to_productions.items():
        new_sym_to_productions[original_symbol] = [
            x for x in prods if x.symbol() in symbols
        ]
        if (
            len(new_sym_to_productions[original_symbol]) == 0
            and not tolerate_pruning_entire_productions
        ):
            raise TypeError(
                f"All productions for {original_symbol} were pruned. "
                f"Check that the target types are correct."
            )
        if original_symbol in stable_symbols:
            continue
        new_sym_to_productions[original_symbol] = Production.reindex(
            new_sym_to_productions[original_symbol]
        )
    return new_sym_to_productions
