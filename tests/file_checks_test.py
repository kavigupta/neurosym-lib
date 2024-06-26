import ast
import glob
import json
import os
import re
import unittest
from functools import lru_cache
from types import ModuleType

import parameterized

import neurosym as ns
from neurosym.examples import near
from tests.tutorial.utils import ipynb_to_py


def files_to_examine(*paths):
    for path in paths:
        for root, _, files in os.walk(path):
            for file in files:
                if ".ipynb_checkpoints" in root:
                    continue
                if file.endswith(".py") or file.endswith(".ipynb"):
                    yield os.path.join(root, file)


def read_python_file(path):
    with open(path) as f:
        text = f.read()
    if path.endswith(".py"):
        return text
    if path.endswith(".ipynb"):
        code = ipynb_to_py(json.loads(text))
        code = "\n".join(code)
        return code
    raise ValueError(f"Unknown file type: {path}")


class GatherImports(ast.NodeVisitor):
    def __init__(self):
        self.imports = set()

    def visit_Import(self, node):
        self.imports.add(node)

    def visit_ImportFrom(self, node):
        self.imports.add(node)


class OnlyDirectImportsTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [(path,) for path in files_to_examine("tests", "tutorial")]
    )
    def test_only_direct_import(self, path):
        code = read_python_file(path)
        tree = ast.parse(code)
        gatherer = GatherImports()
        gatherer.visit(tree)
        imports = {ast.unparse(node) for node in gatherer.imports}
        imports = {imp for imp in imports if "neurosym" in imp}

        expected = {"from neurosym.examples import near", "import neurosym as ns"}

        self.assertEqual(imports | expected, expected)


class NoPrintsTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [(path,) for path in files_to_examine("neurosym")]
    )
    def test_no_prints(self, path):
        if path in {"neurosym/examples/near/methods/near_example_trainer.py"}:
            # skip this file, it is an example
            return
        if path in {"neurosym/utils/logging.py"}:
            # skip this file, it is a logging utility
            return
        with open(path) as f:
            code = f.read()
        code = ast.parse(code)
        nodes = ast.walk(code)
        nodes = [node for node in nodes if isinstance(node, ast.Name)]
        nodes = [node for node in nodes if node.id == "print"]
        self.assertEqual(len(nodes), 0)


@lru_cache(None)
def documented_functions():
    rst_files = glob.glob("docs/source/*.rst")
    functions = set()
    for rst_file in rst_files:
        with open(rst_file) as f:
            text = f.read()
        functions.update(
            x.group("name")
            for x in re.finditer(
                r".. auto(function|class|attribute):: (?P<name>.*)", text
            )
        )
    return functions


class AllFunctionsDocumentedTest(unittest.TestCase):

    pattern = re.compile(r"\b(ns|near)(\.(\w+))+")

    def normalize(self, function):
        if function.startswith("ns."):
            function = function.replace("ns.", "neurosym.", 1)
        if function.startswith("near."):
            function = function.replace("near.", "neurosym.examples.near.", 1)
        function = {
            "neurosym.examples.near.NeuralDSL.from_dsl": "neurosym.examples.near.NeuralDSL",
            "neurosym.PythonDSLSubset.from_s_exps": "neurosym.PythonDSLSubset",
            "neurosym.PythonDSLSubset.from_programs": "neurosym.PythonDSLSubset",
            "neurosym.Environment.empty": "neurosym.Environment",
        }.get(function, function)
        return function

    @parameterized.parameterized.expand(
        [(path,) for path in files_to_examine("tests", "tutorial")]
    )
    def test_all_used_functions_documented(self, path):
        code = read_python_file(path)
        functions = set(x.group() for x in self.pattern.finditer(code))
        functions = {self.normalize(f) for f in functions}
        print(documented_functions())
        extras = functions - documented_functions()
        if not extras:
            return
        self.fail(f"Found {len(extras)} undocumented functions: {extras}")

    def assertFieldDocumented(self, name, base_module, base_module_name):
        if name.startswith("_"):
            return
        if isinstance(getattr(base_module, name), ModuleType):
            return
        self.assertIn(base_module_name + "." + name, documented_functions())

    @parameterized.parameterized.expand([(name,) for name in dir(ns)])
    def test_ns_function_documented(self, name):
        self.assertFieldDocumented(name, ns, "neurosym")

    @parameterized.parameterized.expand([(name,) for name in dir(near)])
    def test_near_function_documented(self, name):
        self.assertFieldDocumented(name, near, "neurosym.examples.near")
