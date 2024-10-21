from neurosym.examples.near.methods.base_trainer import BaseTrainer, BaseTrainerConfig
from neurosym.examples.near.models.generic_mlp_rnn import GenericMLPRNNNeuralHoleFiller
from neurosym.examples.near.validation import (
    UninitializableProgramError,
    ValidationCost,
)

from .dsls import debug_nested_dsl
from .dsls.sequential_differentiable_dsl import example_rnn_dsl
from .dsls.simple_differentiable_dsl import differentiable_arith_dsl
from .interface import NEAR
from .methods.near_example_trainer import (
    NEARTrainer,
    NEARTrainerConfig,
    classification_mse_loss,
)
from .models.mlp import MLP, MLPConfig, mlp_factory
from .models.rnn import (
    RNNConfig,
    Seq2ClassRNN,
    Seq2SeqRNN,
    rnn_factory_seq2class,
    rnn_factory_seq2seq,
)
from .models.torch_program_module import TorchProgramModule
from .models.transformer import (
    BasicMultiDimensionalPositionalEncoding,
    NearTransformer,
    TransformerNeuralHoleFiller,
)
from .neural_dsl import NeuralDSL, PartialProgramNotFoundError, create_modules
from .neural_hole_filler import (
    DictionaryNeuralHoleFiller,
    DoNothingNeuralHoleFiller,
    NeuralHoleFiller,
    UnionNeuralHoleFiller,
)
from .search_graph import near_graph
