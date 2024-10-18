from abc import ABC, abstractmethod
from types import NoneType
from typing import Union

from neurosym.types.type_with_environment import TypeWithEnvironment


class NeuralHoleFiller(ABC):
    @abstractmethod
    def initialize_module(
        self, type_with_environment: TypeWithEnvironment
    ) -> Union[object, NoneType]:
        pass


class DictionaryHoleFiller(NeuralHoleFiller):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def initialize_module(
        self, type_with_environment: TypeWithEnvironment
    ) -> Union[object, NoneType]:
        if type_with_environment.typ not in self.dictionary:
            return None
        return self.dictionary[type_with_environment.typ]()
