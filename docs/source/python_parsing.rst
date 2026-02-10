Python/S-Expression Conversion
===========================================

.. autofunction:: neurosym.s_exp_to_python
.. autofunction:: neurosym.python_to_s_exp
.. autofunction:: neurosym.to_type_annotated_ns_s_exp
.. autofunction:: neurosym.python_to_python_ast
.. autofunction:: neurosym.python_statements_to_python_ast
.. autofunction:: neurosym.python_statement_to_python_ast
.. autofunction:: neurosym.python_to_type_annotated_ns_s_exp
.. autofunction:: neurosym.s_exp_to_python_ast

PythonAST object
------------------------------------------

.. autoclass:: neurosym.PythonAST
    :members:
.. autoclass:: neurosym.NodeAST
.. autoclass:: neurosym.SequenceAST
.. autoclass:: neurosym.ListAST
.. autoclass:: neurosym.LeafAST
.. autoclass:: neurosym.SliceElementAST
.. autoclass:: neurosym.StarrableElementAST
.. autoclass:: neurosym.SpliceAST
.. autoclass:: neurosym.PythonSymbol
    :members: render_symbol

PythonAST factory functions
------------------------------------------

.. autofunction:: neurosym.make_python_ast.make_constant
.. autofunction:: neurosym.make_python_ast.make_name
.. autofunction:: neurosym.make_python_ast.make_call
.. autofunction:: neurosym.make_python_ast.make_expr_stmt
