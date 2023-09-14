from neurosym.dsl.dsl import DSL
from neurosym.dsl.production import ConcreteProduction, ParameterizedProduction
from neurosym.dsl.variable_system import LambdasVariableSystem, NoVariables
from neurosym.types.type_string_repr import TypeDefiner, parse_type
import numpy as np
from neurosym.types.type_signature import expansions
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
        self.variable_system = NoVariables()
        self.max_expansion_steps = max_expansion_steps
        self.max_overall_depth = max_overall_depth

    def typedef(self, key, type_str):
        """
        Define a type.
        """
        self.t.typedef(key, type_str)

    def lambdas(self, lambda_arity_limit: int = 2, num_variable_limit: int = 3):
        self.variable_system = LambdasVariableSystem(
            lambda_arity_limit=lambda_arity_limit, num_variable_limit=num_variable_limit
        )

    def concrete(self, symbol, type_str, fn):
        """
        Add a concrete production to the DSL.
        """
        self._concrete_productions.append(
            (
                symbol,
                self.t.sig(type_str),
                fn,
            )
        )

    def parameterized(self, symbol, type_str, fn, parameters):
        """
        Add a parameterized production to the DSL.
        """
        self._parameterized_productions.append(
            (
                symbol,
                self.t.sig(type_str),
                fn,
                parameters,
            )
        )

    def finalize(self):
        """
        Finalize the DSL.
        """

        # mine the atomic types:
        atomic_types = set()
        num_arrow_args = set()
        has_list = False
        for symbol, sig, fn in self._concrete_productions:
            for t in sig.astype().walk_type_nodes():
                if t.is_atomic():
                    atomic_types.add(t)
                if isinstance(t, ArrowType):
                    num_arrow_args.add(len(t.input_type))
                if isinstance(t, ListType):
                    has_list = True

        for symbol, sig, fn, parameters in self._parameterized_productions:
            for t in sig.astype().walk_type_nodes():
                if t.is_atomic():
                    atomic_types.add(t)
                if isinstance(t, ArrowType):
                    num_arrow_args.add(len(t.input_type))
                if isinstance(t, ListType):
                    has_list = True

        expand_to = []

        fresh_var = 0

        def fresh():
            nonlocal fresh_var
            fresh_var += 1
            return f"#FRESH{fresh_var}"

        if has_list:
            expand_to.append(parse_type(f"[{fresh()}]"))
        for n in num_arrow_args:
            args = [fresh() for _ in range(n)]
            expand_to.append(parse_type(f"({','.join(args)}) -> {fresh()}"))
        for t in atomic_types:
            expand_to.append(t)

        print("concrete productins: ", len(self._concrete_productions))
        print("parameterized productions: ", len(self._parameterized_productions))
        print("expansions to consider:", [render_type(t) for t in expand_to])

        productions = []
        for symbol, sig, fn in self._concrete_productions:
            if not sig.has_type_vars():
                productions.append(ConcreteProduction(symbol, sig, fn))
            else:
                ran_once = False
                for i, expansion in enumerate(
                    expansions(
                        sig,
                        expand_to,
                        max_expansion_steps=self.max_expansion_steps,
                        max_overall_depth=self.max_overall_depth,
                    )
                ):
                    sym = f"{symbol}_{i}"
                    productions.append(ConcreteProduction(sym, expansion, fn))
                    ran_once = True
                assert ran_once, f"No expansions within depth/step bounds for {symbol}"

        for symbol, sig, fn, parameters in self._parameterized_productions:
            if not sig.has_type_vars():
                productions.append(ParameterizedProduction(symbol, sig, fn, parameters))
            else:
                ran_once = False
                for i, expansion in enumerate(
                    expansions(
                        sig,
                        expand_to,
                        max_expansion_steps=self.max_expansion_steps,
                        max_overall_depth=self.max_overall_depth,
                    )
                ):
                    sym = f"{symbol}_{i}"
                    productions.append(
                        ParameterizedProduction(sym, expansion, fn, parameters)
                    )
                    ran_once = True
                assert ran_once, f"No expansions within depth/step bounds for {symbol}"

        for p in productions:
            print(p.render())

        print("productions: ", len(productions))

        return DSL(productions, self.variable_system)
