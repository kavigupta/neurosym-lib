``NEAR`` Algorithm
===========================================

.. autofunction:: neurosym.examples.near.near_graph
.. autofunction:: neurosym.examples.near.validated_near_graph
.. autoclass:: neurosym.examples.near.NeuralDSL
    :members:
.. autoclass:: neurosym.examples.near.NEARTrainerConfig
.. autoclass:: neurosym.examples.near.TorchProgramModule
.. autoclass:: neurosym.examples.near.PartialProgramNotFoundError

``NEAR`` Cost
--------------------------------------------
.. autoclass:: neurosym.examples.near.NearCost
    :members:
.. autoclass:: neurosym.examples.near.default_near_cost
    :members:
.. autoclass:: neurosym.examples.near.NearStructuralCost
    :members:
.. autoclass:: neurosym.examples.near.PerNodeNearStructuralCost
    :members:
.. autoclass:: neurosym.examples.near.NumberHolesNearStructuralCost
    :members:
.. autoclass:: neurosym.examples.near.MinimalStepsNearStructuralCost
    :members:
.. autoclass:: neurosym.examples.near.NearValidationHeuristic
    :members:
.. autoclass:: neurosym.examples.near.ValidationCost
    :members:
.. autoclass:: neurosym.examples.near.ProgramEmbedding
    :members:
.. autoclass:: neurosym.examples.near.IdentityProgramEmbedding
    :members:


``NEAR`` Heirarchical Algorithm
--------------------------------------------
.. autofunction:: neurosym.examples.near.heirarchical_near_graph

NEAR Interface
--------------------------------------------
.. autoclass:: neurosym.examples.near.interface.NEAR
    :members:

NEAR Operations
--------------------------------------------

.. autofunction:: neurosym.examples.near.operations.ite_torch
.. autofunction:: neurosym.examples.near.operations.map_torch
.. autofunction:: neurosym.examples.near.operations.fold_torch

NEAR Factory Functions
--------------------------------------------

.. autoclass:: neurosym.examples.near.NeuralHoleFiller
    :members:
.. autoclass:: neurosym.examples.near.DictionaryNeuralHoleFiller
.. autoclass:: neurosym.examples.near.DoNothingNeuralHoleFiller
.. autoclass:: neurosym.examples.near.UnionNeuralHoleFiller
.. autoclass:: neurosym.examples.near.GenericMLPRNNNeuralHoleFiller
.. autoclass:: neurosym.examples.near.TransformerNeuralHoleFiller
.. autofunction:: neurosym.examples.near.create_modules
.. autofunction:: neurosym.examples.near.mlp_factory
.. autoclass:: neurosym.examples.near.MLPConfig
.. autoclass:: neurosym.examples.near.MLP
.. autofunction:: neurosym.examples.near.rnn_factory_seq2seq
.. autofunction:: neurosym.examples.near.rnn_factory_seq2class
.. autoclass:: neurosym.examples.near.RNNConfig
.. autoclass:: neurosym.examples.near.Seq2SeqRNN
.. autoclass:: neurosym.examples.near.Seq2ClassRNN

NEAR Transformer
--------------------------------------------

.. autoclass:: neurosym.examples.near.NearTransformer
.. autoclass:: neurosym.examples.near.BasicMultiDimensionalPositionalEncoding


Example Datasets/DSLs
--------------------------------------------
.. autofunction:: neurosym.datasets.near_data_example
.. autofunction:: neurosym.examples.near.differentiable_arith_dsl
.. autofunction:: neurosym.examples.near.example_rnn_dsl

Utilites
--------------------------------------------
.. autofunction:: neurosym.examples.near.classification_mse_loss
.. autofunction:: neurosym.examples.near.schedule_optimizer
.. autoclass:: neurosym.examples.near.UninitializableProgramError

For testing and demonstration purposes
--------------------------------------------
.. autofunction:: neurosym.examples.near.debug_nested_dsl.get_combinator_dsl
.. autofunction:: neurosym.examples.near.debug_nested_dsl.get_variable_dsl
.. autofunction:: neurosym.examples.near.debug_nested_dsl.get_dataset
.. autofunction:: neurosym.examples.near.debug_nested_dsl.run_near_on_dsl
