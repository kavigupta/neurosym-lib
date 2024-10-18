from abc import ABC, abstractmethod
from types import NoneType
from typing import Callable, Dict, Union

from torch import nn

from neurosym.types.type import Type
from neurosym.types.type_with_environment import TypeWithEnvironment


class NeuralHoleFiller(ABC):
    """
    Represents an abstract class for filling holes in a program with neural modules.
    """

    @abstractmethod
    def initialize_module(
        self, type_with_environment: TypeWithEnvironment
    ) -> Union[nn.Module, NoneType]:
        """
        Initialize a module for a given TypeWithEnvironment.
        """


class DictionaryHoleFiller(NeuralHoleFiller):
    """
    A hole filler that uses a dictionary to map types to neural modules.
    """

    def __init__(self, dictionary: Dict[Type, Callable[[], nn.Module]]):
        self.dictionary = dictionary

    def initialize_module(
        self, type_with_environment: TypeWithEnvironment
    ) -> Union[nn.Module, NoneType]:
        if type_with_environment.typ not in self.dictionary:
            return None
        return self.dictionary[type_with_environment.typ]()
