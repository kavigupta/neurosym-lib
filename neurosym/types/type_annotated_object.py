from dataclasses import dataclass

from .type import Type


@dataclass
class TypeAnnotatedObject:
    """
    An object that has a type annotation. This is used to pass around objects
    in environments, with their types.

    :param object_type: The type of the object.
    :param object_value: The object itself.
    """

    object_type: Type
    object_value: object
