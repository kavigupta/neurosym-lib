import ast
from functools import lru_cache
import glob
import json
import os
import re
import unittest

import parameterized

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
            for x in re.finditer(r".. (autofunction|autoclass):: (?P<name>.*)", text)
        )
    return functions


class AllFunctionsDocumentedTest(unittest.TestCase):

    pattern = re.compile(r"\b(ns|near)(\.(\w+))+")

    def normalize(self, function):
        return function.replace("ns.", "neurosym.").replace(
            "near.", "neurosym.examples.near."
        )

    @parameterized.parameterized.expand(
        [(path,) for path in files_to_examine("tests", "tutorial")]
    )
    def test_all_functions_dcoumented(self, path):
        code = read_python_file(path)
        functions = set(x.group() for x in self.pattern.finditer(code))
        functions = {self.normalize(f) for f in functions}
        print(documented_functions())
        extras = functions - documented_functions()
        if not extras:
            return
        self.fail(f"Found {len(extras)} undocumented functions: {extras}")