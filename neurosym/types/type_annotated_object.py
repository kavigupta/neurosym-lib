from dataclasses import dataclass

from .type import Type


@dataclass
class TypeAnnotatedObject:
    """
    An object that has a type annotation.
    """

    object_type: Type
    object_value: object
