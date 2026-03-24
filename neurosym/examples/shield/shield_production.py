from dataclasses import dataclass
from typing import Dict

from neurosym.dsl.dsl_factory import DSLFactory
from neurosym.dsl.production import Production
from neurosym.types.type_signature import TypeSignature

from .shield_type_signature import ShieldTypeSignature


def add_shield_productions(dslf: DSLFactory):
    """
    Add shield productions to a DSLFactory. Must be called after
    the factory is constructed (uses ``max_env_depth``).

    :param dslf: The DSLFactory to add shield productions to.
    """
    dslf.extra_productions(
        "<shield>",
        [
            ShieldProduction(ShieldTypeSignature(i))
            for i in range(dslf.max_env_depth)
        ],
    )


@dataclass
class ShieldProduction(Production):
    """
    This production represents a shield operation.

    `shieldk` removes variable `k` from the environment.

    :param _type_signature: the type signature of this shield production
    """

    _type_signature: ShieldTypeSignature

    def base_symbol(self):
        return f"shield{self._type_signature.index_in_env}"

    def get_index(self):
        return None

    def with_index(self, index):
        assert index == 0
        return ShieldProduction(self._type_signature)

    def type_signature(self) -> TypeSignature:
        return self._type_signature

    def initialize(self, dsl) -> Dict[str, object]:
        del dsl
        return {}

    def apply(self, dsl, state, children, environment):
        assert len(children) == 1
        return dsl.compute(
            children[0],
            environment[: self._type_signature.index_in_env]
            + environment[self._type_signature.index_in_env + 1 :],
        )

    def render(self):
        return f"{self.symbol():>15} :: {self._type_signature.render()}"
