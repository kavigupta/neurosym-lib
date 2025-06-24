from abc import ABC, abstractmethod
from dataclasses import KW_ONLY, dataclass
from typing import Callable, Dict, Union

from torch import NoneType

from ..types.type_signature import (
    LambdaTypeSignature,
    TypeSignature,
    VariableTypeSignature,
)


class Production(ABC):
    """
    Represents a production rule in a simple s-expression grammar.

    Has a symbol, a type signature, and a function that computes the
    semantics of the production.
    """

    @abstractmethod
    def base_symbol(self) -> str:
        """
        Get the "base" symbol of this production, without the index.
        """

    @abstractmethod
    def get_index(self) -> Union[int, NoneType]:
        """
        Get the index of this production; this is used to disambiguate
        productions with the same base symbol. These are created by
        the :py:class:`neurosym.dsl.DSLFactory` either by
        :py:meth:`neurosym.dsl.DSLFactory.lambdas` or by
        type variables that appear on only the left or right side of a type signature
        being enumerated as templates.
        """

    def get_numerical_index(self) -> int:
        """
        Like :py:meth:`get_index`, but returns 0 if the index is None.
        """
        idx = self.get_index()
        return 0 if idx is None else idx

    @abstractmethod
    def with_index(self, index: int) -> "Production":
        """
        A copy of this production with the given index.
        """

    def symbol(self) -> str:
        """
        Return the symbol of this production. This is
        the base symbol if the index is None, otherwise
        it is the base symbol followed by an underscore and the index.
        """
        if self.get_index() is None:
            return self.base_symbol()
        return f"{self.base_symbol()}_{self.get_index()}"

    @abstractmethod
    def type_signature(self) -> TypeSignature:
        """
        Return a TypeSignature object representing the type signature of this production
        """

    @abstractmethod
    def initialize(self, dsl) -> Dict[str, object]:
        """
        Return some state that this production might need to compute its function.
        E.g., for a neural network production, this might be the weights of the network.
        """

    @abstractmethod
    def apply(self, dsl, state, children, environment):
        """
        Apply this production to the given children.

        :param dsl: the DSL to apply relative to
        :param state: the state dictionary for this production
        :param children: the children of this production, as InitializedSExpression objects
        :param environment: the current de-bruijn environment. environment[i] corresponds to variable i.
        """

    @abstractmethod
    def render(self) -> str:
        """
        Render this production as a string.
        """

    @classmethod
    def reindex(cls, productions):
        """
        Reindex the given productions, so that they are indexed from 0 to len(productions) - 1.
        """
        if len(productions) == 1:
            return [productions[0].with_index(None)]
        return [p.with_index(i) for i, p in enumerate(productions)]


class FunctionLikeProduction(Production):
    """
    Production that acts like a function, i.e., evaluates each of its children
    as values and then applies a function to them.
    """

    @abstractmethod
    def evaluate(self, dsl, state, inputs, environment):
        """
        Return the resulting pytorch expression of computing this function on the inputs
        Takes in the state of the production, which is the result of
        :py:meth:`neurosym.DSL.initialize`

        Effectively a form of denotation semantics.
        """

    def apply(self, dsl, state, children, environment):
        return self.evaluate(
            dsl, state, [dsl.compute(x, environment) for x in children], environment
        )


@dataclass
class ConcreteProduction(FunctionLikeProduction):
    """
    Represents a production rule in a simple s-expression grammar, that does
    not have any parameters. This is added to the DSL by the :py:class:`neurosym.DSLFactory`
    when :py:meth:`neurosym.DSLFactory.concrete` is called.
    """

    _symbol: str
    _type_signature: TypeSignature
    _compute: Callable[..., object]
    _: KW_ONLY
    index: int = None

    def base_symbol(self):
        return self._symbol

    def get_index(self):
        return self.index

    def with_index(self, index):
        return type(self)(
            self._symbol, self._type_signature, self._compute, index=index
        )

    def type_signature(self) -> TypeSignature:
        return self._type_signature

    def initialize(self, dsl) -> Dict[str, object]:
        del dsl
        return {}

    def evaluate(self, dsl, state, inputs, environment):
        del dsl, environment
        assert state == {}
        try:
            return self._compute(*inputs)
        except TypeError as exc:
            raise TypeError(
                f"Error computing {self.symbol()} on inputs {inputs} with state {state}"
            ) from exc

    def render(self):
        return f"{self.symbol():>15} :: {self._type_signature.render()}"


@dataclass
class LambdaProduction(Production):
    """
    This production represents a lambda function. This is added automatically
    to the DSL by the :py:class:`neurosym.DSLFactory` when :py:meth:`neurosym.DSLFactory.lambdas`
    is called.
    """

    _unique_id: int
    _type_signature: LambdaTypeSignature

    @property
    def arity(self):
        return self._type_signature.function_arity()

    def base_symbol(self):
        return "lam"

    def get_index(self):
        return self._unique_id

    def with_index(self, index):
        return LambdaProduction(index, self._type_signature)

    def type_signature(self) -> TypeSignature:
        return self._type_signature

    def initialize(self, dsl) -> Dict[str, object]:
        del dsl
        return {}

    def apply(self, dsl, state, children, environment):
        """
        Apply the lambda production to its child, creating
        a function with the child as the body.
        """
        # pylint: disable=cyclic-import
        from neurosym.dsl.lambdas import LambdaFunction

        [body] = children
        return LambdaFunction.of(dsl, body, self._type_signature, environment)

    def render(self):
        return f"{self.symbol():>15} :: {self._type_signature.render()}"


@dataclass
class VariableProduction(Production):
    """
    A production that represents a de Bruijn variable. This is added automatically
    to the DSL by the :py:class:`neurosym.DSLFactory` when
    :py:meth:`neurosym.DSLFactory.lambdas` is called.
    """

    _unique_id: int
    _type_signature: VariableTypeSignature

    def base_symbol(self):
        return f"${self._type_signature.index_in_env}"

    def get_index(self):
        return self._unique_id

    def with_index(self, index):
        return VariableProduction(index, self._type_signature)

    def type_signature(self) -> TypeSignature:
        return self._type_signature

    def initialize(self, dsl) -> Dict[str, object]:
        del dsl
        return {}

    def apply(self, dsl, state, children, environment):
        del dsl, state, children
        return environment[self._type_signature.index_in_env].object_value

    def render(self):
        return f"{self.symbol():>15} :: {self._type_signature.render()}"

    @property
    def index_in_env(self):
        return self._type_signature.index_in_env


@dataclass
class ParameterizedProduction(ConcreteProduction):
    """
    Like a concrete production, but with some parameters that need to be initialized.

    This is added to the DSL by the :py:class:`neurosym.DSLFactory` when
    :py:meth:`neurosym.DSLFactory.parameterized` is called.
    """

    initializers: Dict[str, Callable[[], object]]
    provide_enviroment: Union[NoneType, str] = None

    def with_index(self, index):
        # pylint: disable=unexpected-keyword-arg
        return ParameterizedProduction(
            self._symbol,
            self._type_signature,
            self._compute,
            index=index,
            initializers=self.initializers,
        )

    def initialize(self, dsl) -> Dict[str, object]:
        del dsl
        return {k: v() for k, v in self.initializers.items()}

    def evaluate(self, dsl, state, inputs, environment):
        del dsl
        kwargs = state.copy()
        if self.provide_enviroment is not None:
            kwargs[self.provide_enviroment] = environment
        return self._compute(*inputs, **kwargs)

    def render(self):
        lhs = f"{self.symbol()}[{', '.join(self.initializers)}]"
        return f"{lhs:>15} :: {self._type_signature.render()}"
