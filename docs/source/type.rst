Types
===========================================

.. autoclass:: neurosym.Type
    :members:

.. autofunction:: neurosym.parse_type
.. autofunction:: neurosym.render_type
.. autoclass:: neurosym.TypeDefiner
    :members: __call__, sig, typedef, filtered_type_variable, lookup_type, lookup_filter
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
.. autoclass:: neurosym.GenericTypeVariable
.. autoclass:: neurosym.TypeVariable
.. autoclass:: neurosym.FilteredTypeVariable

Type Signatures
------------------------------------------

.. autoclass:: neurosym.TypeSignature
    :members:
.. autoclass:: neurosym.FunctionTypeSignature
    :members: astype
.. autoclass:: neurosym.VariableTypeSignature
.. autoclass:: neurosym.LambdaTypeSignature
    :members: function_arity

Other helpful functions
------------------------------------------
.. autofunction:: neurosym.lex_type
.. autofunction:: neurosym.type_expansions
.. autofunction:: neurosym.bottom_up_enumerate_types