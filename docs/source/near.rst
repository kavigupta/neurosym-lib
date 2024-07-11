``NEAR`` Algorithm
===========================================

.. autofunction:: neurosym.examples.near.near_graph
.. autoclass:: neurosym.examples.near.NeuralDSL
    :members:
.. autoclass:: neurosym.examples.near.NEARTrainerConfig
.. autoclass:: neurosym.examples.near.NEARTrainer
.. autoclass:: neurosym.examples.near.ValidationCost
    :members:
.. autoclass:: neurosym.examples.near.TorchProgramModule
.. autoclass:: neurosym.examples.near.PartialProgramNotFoundError

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

.. autofunction:: neurosym.examples.near.create_modules
.. autofunction:: neurosym.examples.near.mlp_factory
.. autoclass:: neurosym.examples.near.MLPConfig
.. autoclass:: neurosym.examples.near.MLP
.. autofunction:: neurosym.examples.near.rnn_factory_seq2seq
.. autofunction:: neurosym.examples.near.rnn_factory_seq2class
.. autoclass:: neurosym.examples.near.RNNConfig
.. autoclass:: neurosym.examples.near.Seq2SeqRNN
.. autoclass:: neurosym.examples.near.Seq2ClassRNN

Example Datasets/DSLs
--------------------------------------------
.. autofunction:: neurosym.datasets.near_data_example
.. autofunction:: neurosym.examples.near.differentiable_arith_dsl
.. autofunction:: neurosym.examples.near.example_rnn_dsl

Utilites
--------------------------------------------
.. autofunction:: neurosym.examples.near.classification_mse_loss
.. autoclass:: neurosym.examples.near.BaseTrainerConfig
.. autoclass:: neurosym.examples.near.BaseTrainer
.. autoclass:: neurosym.examples.near.UninitializableProgramError