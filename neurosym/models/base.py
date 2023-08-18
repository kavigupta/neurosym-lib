from dataclasses import dataclass
from torch import nn

@dataclass
class BaseConfig:
    model_name: str # Provides a model-speicfic identification token in the SExpression.