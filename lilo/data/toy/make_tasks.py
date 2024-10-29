"""
toy : make_tasks.py | Author : Sagnik Anupam.

Utility functions for loading tasks for the math domain. This domain was designed in the "Neurosymbolic Reasoning for Mathematical Domains" project and builds on the mathematical domain in the Lemma and ConPoLe papers.
"""

import os
import random
from src.task_loaders import *
from data.math.grammar import *

import dreamcoder.domains.math.makeMathTasks as math_legacy
from dreamcoder.task import Task
from dreamcoder.type import *

DOMAIN_NAME = "toy"
ROOT_DIR = os.getcwd()
DEFAULT_DATA_DIRECTORY = os.path.join(ROOT_DIR, f"dreamcoder/data/{DOMAIN_NAME}")
TASKS = ""
DEFAULT_TASKS_DIRECTORY = os.path.join(DEFAULT_DATA_DIRECTORY, TASKS)

@TaskLoaderRegistry.register
class ToyLoader(TaskDataLoader):
    name = "toy"
    def addN(n):
        x = random.choice(range(500))
        return {"i": x, "o": x + n}
    
    def add1(): return addN(1)
    def add2(): return addN(2)
    def add3(): return addN(3)
    
    def load_tasks(self):
        tasks = {"train": [], "test": []}
        request = arrow(tint, tint)
        training_examples = [
        {"name": "add1", "examples": [add1() for _ in range(5000)]},
        {"name": "add2", "examples": [add2() for _ in range(5000)]},
        {"name": "add3", "examples": [add3() for _ in range(5000)]},
        ]
        for split in tasks.keys():
            tasks[split] = [
                    Task(
                        name=task["name"],
                        request=request,
                        examples= [((ex["i"],), ex["o"]) for ex in task["examples"]],
                        features=None,
                        cache=False,
                    )
                    for task in training_examples
                ]
        return {TRAIN: tasks["train"], TEST: tasks["test"]}
    

@TaskLanguageLoaderRegistry.register
class ToyLanguageLoader(TaskDataLoader):
    name = "toy"

    def load_task_language(self):
        return ({},{}) # No language for math tasks
    

