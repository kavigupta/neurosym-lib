import ast
import functools
import inspect
import json
import os
import subprocess
import unittest
from functools import lru_cache
from inspect import isfunction, ismethod
from types import ModuleType

import parameterized
import sphobjinv

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


def all_functions_in_module(module):
    if not module.__name__.startswith("neurosym"):
        return
    for name in dir(module):
        if name.startswith("_"):
            continue
        if isinstance(getattr(module, name), ModuleType):
            yield from all_functions_in_module(getattr(module, name))
        if isinstance(getattr(module, name), type):
            yield from all_functions_in_class(getattr(module, name))
        elif isfunction(getattr(module, name)):
            yield from all_functions_in_function(getattr(module, name))


def all_functions_in_class(cls):
    if not cls.__module__.startswith("neurosym"):
        return
    yield cls
    for name in dir(cls):
        if name.startswith("_"):
            continue
        if ismethod(getattr(cls, name)):
            yield from all_functions_in_method(getattr(cls, name))
        elif isfunction(getattr(cls, name)):
            yield from all_functions_in_function(getattr(cls, name))


def all_functions_in_method(method):
    if not method.__class__.__module__.startswith("neurosym"):
        return
    yield method


def all_functions_in_function(function):
    if not function.__module__.startswith("neurosym"):
        return
    yield function


def lookup_qualified_name(base, qualified_chunks):
    if not qualified_chunks:
        return base
    return lookup_qualified_name(
        getattr(base, qualified_chunks[0]), qualified_chunks[1:]
    )


def lookup_qualified_neurosym_name(qualified_name):
    if qualified_name.startswith("neurosym."):
        return lookup_qualified_name(ns, qualified_name.split(".")[1:])
    raise ValueError(f"Unknown module: {qualified_name}")


@lru_cache(None)
def read_obj_inv():
    path = "docs/build/html/objects.inv"

    inv = sphobjinv.Inventory(path)
    return [
        lookup_qualified_neurosym_name(x.name)
        for x in inv.objects
        if x.name.startswith("neurosym")
    ]


def get_objects():
    objs = [
        obj
        for module in (ns, near)
        for obj in all_functions_in_module(module)
        if not getattr(obj, "__internal_only__", False)
    ]
    unique_objects = []
    for obj in objs:
        if obj not in unique_objects:
            unique_objects.append(obj)
    return unique_objects


objects = get_objects()


class AllImplicitlyReferencedFunctionsDocumentedTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        subprocess.run(["make", "html"], cwd="docs", check=True)

    @parameterized.parameterized.expand([(i,) for i in range(len(objects))])
    def test_documented(self, i):
        obj = objects[i]
        print(obj)
        if obj in read_obj_inv():
            return
        if self.is_inherited_and_undocumented(obj):
            return

        self.fail(f"Object {obj} not documented")

    def is_inherited_and_undocumented(self, obj):
        if not isfunction(obj):
            return False
        if hasattr(obj, "__doc__") and obj.__doc__ is not None:
            return False
        clas = get_class_that_defined_method(obj)
        if clas is None:
            return False
        for base in clas.__bases__:
            if obj.__name__ in dir(base):
                return True
        return False


def get_class_that_defined_method(meth):
    if isinstance(meth, functools.partial):
        return get_class_that_defined_method(meth.func)
    if inspect.ismethod(meth) or (
        inspect.isbuiltin(meth)
        and getattr(meth, "__self__", None) is not None
        and getattr(meth.__self__, "__class__", None)
    ):
        for cls in inspect.getmro(meth.__self__.__class__):
            if meth.__name__ in cls.__dict__:
                return cls
        meth = getattr(meth, "__func__", meth)  # fallback to __qualname__ parsing
    if inspect.isfunction(meth):
        cls = getattr(
            inspect.getmodule(meth),
            meth.__qualname__.split(".<locals>", 1)[0].rsplit(".", 1)[0],
            None,
        )
        if isinstance(cls, type):
            return cls
    return getattr(meth, "__objclass__", None)  # handle special descriptor objects
