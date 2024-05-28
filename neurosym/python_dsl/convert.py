from neurosym.python_dsl.python_ast import PythonAST


def python_to_s_exp(code, **kwargs):
    """
    Converts python code to an s-expression.
    """
    return PythonAST.parse_python_module(code).to_s_exp(**kwargs)


def s_exp_to_python(code):
    """
    Converts an s expression to python code.
    """
    return PythonAST.parse_s_expression(code).to_python()
