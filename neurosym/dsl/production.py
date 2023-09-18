from abc import ABC, abstractmethod
from dataclasses import KW_ONLY, dataclass
from typing import Callable, Dict


from ..types.type_signature import TypeSignature


class Production(ABC):
    """
    Represents a production rule in a simple s-expression grammar.

    Has a symbol, a type signature, and a function that represents the fn represented.
    """

    @abstractmethod
    def base_symbol(self):
        pass

    @abstractmethod
    def get_index(self):
        pass

    @abstractmethod
    def with_index(self, index):
        pass

    def symbol(self):
        """
        Return the symbol of this production.
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
    def apply(self, dsl, state, children):
        """
        Apply this production to the given children.
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
    @abstractmethod
    def evaluate(self, dsl, state, inputs):
        """
        Return the resulting pytorch expression of computing this function on the inputs
            Takes in the state of the production, which is the result of initialize().

        Effectively a form of denotation semantics.
        """

    def apply(self, dsl, state, children):
        """
        Apply this production to the given children.
        """
        return self.evaluate(dsl, state, [dsl.compute(x) for x in children])


@dataclass
class ConcreteProduction(FunctionLikeProduction):
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

    def evaluate(self, dsl, state, inputs):
        del dsl
        assert state == {}
        try:
            return self._compute(*inputs)
        except TypeError:
            raise TypeError(
                f"Error computing {self.symbol()} on inputs {inputs} with state {state}"
            )

    def render(self):
        return f"{self.symbol():>15} :: {self._type_signature.render()}"


@dataclass
class ParameterizedProduction(ConcreteProduction):
    initializers: Dict[str, Callable[[], object]]

    def with_index(self, index):
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

    def evaluate(self, dsl, state, inputs):
        del dsl
        return self._compute(*inputs, **state)

    def render(self):
        lhs = f"{self.symbol()}[{', '.join(self.initializers)}]"
        return f"{lhs:>15} :: {self._type_signature.render()}"
