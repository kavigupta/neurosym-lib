Types
===========================================

.. autoclass:: neurosym.Type
    :members:

.. autofunction:: neurosym.parse_type
.. autofunction:: neurosym.render_type
.. autoclass:: neurosym.TypeDefiner
.. autoclass:: neurosym.Environment
    :members:
.. autoclass:: neurosym.PermissiveEnvironmment
    :members:
.. autoclass:: neurosym.TypeWithEnvironment
    :members:
.. autoclass:: neurosym.UnificationError

Subclasses of :class:`Type`
------------------------------------------

.. autoclass:: neurosym.AtomicType
.. autoclass:: neurosym.TensorType
.. autoclass:: neurosym.ListType
.. autoclass:: neurosym.ArrowType
.. autoclass:: neurosym.TypeVariable
.. autoclass:: neurosym.FilteredTypeVariable

Type Signatures
------------------------------------------

.. autoclass:: neurosym.TypeSignature
    :members:
.. autoclass:: neurosym.FunctionTypeSignature
.. autoclass:: neurosym.VariableTypeSignature
.. autoclass:: neurosym.LambdaTypeSignature
    :members: function_arity

Other helpful functions
------------------------------------------
.. autofunction:: neurosym.lex_type
.. autofunction:: neurosym.type_expansions
.. autofunction:: neurosym.bottom_up_enumerate_types