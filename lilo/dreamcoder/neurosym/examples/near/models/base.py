from dataclasses import dataclass

from neurosym.utils.documentation import internal_only


@internal_only
@dataclass
class BaseConfig:
    model_name: (
        str  # Provides a model-speicfic identification token in the SExpression.
    )
