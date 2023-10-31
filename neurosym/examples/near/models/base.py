from dataclasses import dataclass


@dataclass
class BaseConfig:
    model_name: str  # Provides a model-speicfic identification token in the SExpression.
