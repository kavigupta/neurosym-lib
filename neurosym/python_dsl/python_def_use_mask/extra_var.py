import re
from dataclasses import dataclass

from neurosym.python_dsl.names import PYTHON_DSL_SEPARATOR


@dataclass(frozen=True, eq=True, order=True)
class ExtraVar:
    """
    Used to represent an extra variable, not found in the tree distribution.
    Used in handling De Bruijn variables.

    :param id: The id of the extra variable.
    """

    id: int

    @classmethod
    def from_name(cls, name):
        """
        Compute an extra variable from a name.

        :param name: The name to compute the extra variable from. Should be of the form ``const-&__0:0~Name`` or similar.
        """
        mat = canonicalized_python_name_leaf_regex.match(name)
        if mat:
            return cls(int(mat.group("var")))
        return None

    def leaf_name(self):
        """
        Converts the extra variable to a leaf name.
        """
        # Using typ by default is a bit of a hack, since we should really be using use_type
        # however, this would require us to add a leaf for every version of use_type to the
        # tree distribution
        return canonicalized_python_name_as_leaf(self.id, "Name")


canonicalized_python_name_leaf_regex = re.compile(
    r"const-&(__(?P<var>\d+)):[0-9]+(" + re.escape(PYTHON_DSL_SEPARATOR) + r"[A-Za-z])?"
)


def canonicalized_python_name_as_leaf(name, typ):
    """
    Get the canonicalized python name as a leaf node. E.g., ``const-&__0:0~Name``

    :param name: The name to get the canonicalized python name for.
    :param typ: The type of the name, e.g., Name, NameStr, etc.
    """
    result = f"const-&{canonicalized_python_name(name)}:0"
    result += PYTHON_DSL_SEPARATOR + typ
    return result


def canonicalized_python_name(name):
    """
    Produce a canonical python variable name for an index. E.g., ``__0``, ``__1``, etc.

    :param name: The index to produce the name for.
    """
    return f"__{name}"
