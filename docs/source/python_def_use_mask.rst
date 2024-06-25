Python Def-Use Chain Preorder Mask
===========================================

.. autoclass:: neurosym.python_def_use_mask.DefUseChainPreorderMask
    :members:
.. autoclass:: neurosym.python_def_use_mask.DefUseMaskConfiguration
    :members:

.. autofunction:: neurosym.python_def_use_mask.match_either_name_or_global

.. autoclass:: neurosym.python_def_use_mask.SpecialCaseSymbolPredicate
    :members:

.. autoclass:: neurosym.python_def_use_mask.NameEPredicate

Handlers for Python Def-Use Chain Preorder Mask
------------------------------------------------

.. autoclass:: neurosym.python_def_use_mask.Handler
    :members:
.. autoclass:: neurosym.python_def_use_mask.ConstructHandler
.. autoclass:: neurosym.python_def_use_mask.TargetHandler
.. autoclass:: neurosym.python_def_use_mask.DefaultHandler
.. autoclass:: neurosym.python_def_use_mask.DefiningStatementHandler
.. autoclass:: neurosym.python_def_use_mask.DefiningConstructHandler

.. autofunction:: neurosym.python_def_use_mask.default_handler
.. autofunction:: neurosym.python_def_use_mask.create_target_handler
.. autoclass:: neurosym.python_def_use_mask.HandlerPuller
    :members:

Extra Variables
------------------------------------------------

.. autoclass:: neurosym.python_def_use_mask.ExtraVar
    :members:

.. autofunction:: neurosym.python_def_use_mask.canonicalized_python_name
.. autofunction:: neurosym.python_def_use_mask.canonicalized_python_name_as_leaf