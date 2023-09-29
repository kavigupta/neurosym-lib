from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict
import uuid
import numpy as np


class Type(ABC):
    @abstractmethod
    def walk_type_nodes(self):
        """
        Walk the type tree and yield all nodes.
        """
        raise NotImplementedError

    def node_summary(self):
        """
        Return a summary of the node.
        """
        summary_dict = {
            "_type": self.__class__.__name__,
            **self.inherent_parameters(),
        }
        summary_dict = {k: v for k, v in sorted(summary_dict.items())}
        return str(summary_dict)

    @abstractmethod
    def inherent_parameters(self):
        """
        Return a list of inherent parameters of the type, i.e., all parameters not associated
            with the children.
        """
        raise NotImplementedError

    @abstractmethod
    def children(self):
        """
        Iterate over direct children and yield each.
        """
        raise NotImplementedError

    @abstractmethod
    def unify(
        self, other: "Type", already_tried_other_direction=False
    ) -> Dict[str, "Type"]:
        """
        Unify this type with another type. Returns a dictionary of substitutions.

        Raise UnificationError if the types cannot be unified.

        If already_tried_other_direction is True, then we have already tried to unify
        the other direction and failed, so we should not try again.
        """
        raise NotImplementedError

    @abstractmethod
    def subst_type_vars(self, subst: Dict[str, "Type"]):
        """
        Substitute type variables in the type according to the substitution.
        """
        raise NotImplementedError

    def is_atomic(self):
        """
        Return True if the type is atomic.
        """
        return len(list(self.walk_type_nodes())) == 1 and not isinstance(
            self, TypeVariable
        )

    def depth_of_type_variables(self) -> List[Tuple[str, float]]:
        """
        Return a list of tuples of the form (type_variable, depth) where depth is the
        depth of the type variable in the type tree.
        """
        node_height = np.log2(1 + len(list(self.children())))
        return [
            (tv, depth + node_height)
            for child in self.children()
            for tv, depth in child.depth_of_type_variables()
        ]

    def max_depth_per_type_variable(self) -> Dict[str, float]:
        """
        Return a dictionary mapping type variables to their maximum depth in the type
        tree.
        """
        result = {}
        for tv, depth in self.depth_of_type_variables():
            result[tv] = max(result.get(tv, 0), depth)
        return result

    def get_type_vars(self):
        """
        Return a set of all type variables in the type.
        """
        return sorted({tv for tv, _ in self.depth_of_type_variables()})

    def has_type_vars(self):
        """
        Return True if the type has type variables.
        """
        return len(self.get_type_vars()) > 0

    def depth(self):
        """
        Return the depth of the type tree.
        """
        if not hasattr(self, "_depth"):
            children = [child.depth() for child in self.children()]
            depth = max(children + [0]) + np.log2(len(children) + 1)
            object.__setattr__(self, "_depth", depth)

        return self._depth


class UnificationError(Exception):
    pass


@dataclass(frozen=True, eq=True)
class AtomicType(Type):
    name: str

    def __post_init__(self):
        assert self.name.isidentifier(), f"{self.name} is not a valid identifier"

    def walk_type_nodes(self):
        yield self

    def inherent_parameters(self):
        return dict(name=self.name)

    def children(self):
        yield from []

    def unify(
        self, other: "Type", already_tried_other_direction=False
    ) -> Dict[str, "Type"]:
        if isinstance(other, AtomicType):
            if self.name == other.name:
                return {}
            else:
                raise UnificationError(f"{self} != {other}")
        elif already_tried_other_direction:
            raise UnificationError(f"{self} != {other}")
        else:
            return other.unify(self, already_tried_other_direction=True)

    def subst_type_vars(self, subst: Dict[str, Type]):
        return self


@dataclass(frozen=True, eq=True)
class TensorType(Type):
    """
    A tensor is a type that represents a tensor of a given shape.
    """

    dtype: Type
    shape: Tuple[int]

    def walk_type_nodes(self):
        # is atomic. we do not consider the dtype and shape as nodes.
        yield self

    def inherent_parameters(self):
        return dict(dtype=self.dtype, shape=self.shape)

    def children(self):
        yield from []

    def subst_type_vars(self, subst: Dict[str, Type]):
        assert not self.has_type_vars()
        return self

    def unify(
        self, other: "Type", already_tried_other_direction=False
    ) -> Dict[str, "Type"]:
        if isinstance(other, TensorType):
            if self.shape == other.shape:
                return self.dtype.unify(other.dtype)
            else:
                raise UnificationError(f"{self} != {other}")
        elif already_tried_other_direction:
            raise UnificationError(f"{self} != {other}")
        else:
            return other.unify(self, already_tried_other_direction=True)


@dataclass(frozen=True, eq=True)
class ListType(Type):
    """
    A list type is a type of the form [t] where t is a type.
    """

    element_type: Type

    def walk_type_nodes(self):
        yield self
        yield from self.element_type.walk_type_nodes()

    def inherent_parameters(self):
        return dict()

    def children(self):
        yield self.element_type

    def subst_type_vars(self, subst: Dict[str, Type]):
        return ListType(self.element_type.subst_type_vars(subst))

    def unify(
        self, other: "Type", already_tried_other_direction=False
    ) -> Dict[str, "Type"]:
        if isinstance(other, ListType):
            return self.element_type.unify(other.element_type)
        elif already_tried_other_direction:
            raise UnificationError(f"{self} != {other}")
        else:
            return other.unify(self, already_tried_other_direction=True)


@dataclass(frozen=True, eq=True)
class ArrowType(Type):
    """
    An arrow type is a type of the form t1 -> t2 where t1 and t2 are types.
    """

    input_type: Tuple[Type]
    output_type: Type

    def __post_init__(self):
        assert isinstance(self.input_type, tuple), "input_type must be a tuple"

    def subst_type_vars(self, subst: Dict[str, Type]):
        return ArrowType(
            tuple(t.subst_type_vars(subst) for t in self.input_type),
            self.output_type.subst_type_vars(subst),
        )

    def walk_type_nodes(self):
        yield self
        for t in self.input_type:
            yield from t.walk_type_nodes()
        yield from self.output_type.walk_type_nodes()

    def inherent_parameters(self):
        return dict()

    def children(self):
        yield from self.input_type
        yield self.output_type

    def unify(
        self, other: "Type", already_tried_other_direction=False
    ) -> Dict[str, "Type"]:
        if not isinstance(other, ArrowType):
            if already_tried_other_direction:
                raise UnificationError(f"{self} != {other}")
            else:
                return other.unify(self, already_tried_other_direction=True)
        if len(self.input_type) != len(other.input_type):
            raise UnificationError(f"{self} != {other}")
        individuals = []
        for t1, t2 in zip(self.input_type, other.input_type):
            individuals.append(t1.unify(t2))
        individuals.append(self.output_type.unify(other.output_type))
        subst = {}
        for individual in individuals:
            for k, v in individual.items():
                if k in subst:
                    if subst[k] != v:
                        raise UnificationError(f"{k} is unified to {subst[k]} and {v}")
                else:
                    subst[k] = v
        return subst


@dataclass(frozen=True, eq=True)
class TypeVariable(Type):
    name: str

    @classmethod
    def fresh(cls):
        return TypeVariable(f"_{uuid.uuid4().hex}")

    def __post_init__(self):
        assert self.name.isidentifier(), f"{self.name} is not a valid identifier"

    def depth_of_type_variables(self) -> List[Tuple[str, int]]:
        return [(self.name, 0)]

    def subst_type_vars(self, subst: Dict[str, Type]):
        return subst.get(self.name, self)

    def walk_type_nodes(self):
        yield self

    def inherent_parameters(self):
        return dict()

    def children(self):
        yield from []

    def unify(
        self, other: "Type", already_tried_other_direction=False
    ) -> Dict[str, "Type"]:
        return {self.name: other}


float_t = AtomicType("f")
