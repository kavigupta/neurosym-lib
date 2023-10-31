import ast
import os
import unittest

import parameterized


def files_to_examine():
    for root, _, files in os.walk("tests"):
        for file in files:
            if file.endswith(".py"):
                yield os.path.join(root, file)


class GatherImports(ast.NodeVisitor):
    def __init__(self):
        self.imports = set()

    def visit_Import(self, node):
        self.imports.add(node)

    def visit_ImportFrom(self, node):
        self.imports.add(node)


class OnlyDirectImportsTest(unittest.TestCase):
    @parameterized.parameterized.expand([(path,) for path in files_to_examine()])
    def test_only_direct_import(self, path):
        with open(path) as f:
            code = f.read()
        tree = ast.parse(code)
        gatherer = GatherImports()
        gatherer.visit(tree)
        imports = {ast.unparse(node) for node in gatherer.imports}
        imports = {imp for imp in imports if "neurosym" in imp}

        expected = {"from neurosym.examples import near", "import neurosym as ns"}

        self.assertEqual(imports | expected, expected)
