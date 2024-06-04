import re

from neurosym.python_dsl.names import PYTHON_DSL_SEPARATOR

NAME_REGEX = re.compile(
    r"const-(?P<typ>&)(?P<name>\w+|\*):(?P<scope>\d+)"
    + PYTHON_DSL_SEPARATOR
    + r"(?P<dfa_sym>Name|NullableName|NameStr|NullableNameStr)$"
)
GLOBAL_REGEX = re.compile(
    r"const-(?P<typ>g)_(?P<name>\w+|\*)~(Name|NullableName|NameStr|NullableNameStr)"
)


def match_either_name_or_global(s):
    return NAME_REGEX.match(s) or GLOBAL_REGEX.match(s)
