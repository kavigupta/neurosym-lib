"""
list : make_tasks.py | Author : Sagnik Anupam.

Utility functions for loading tasks for the list domain from DreamCoder.
"""

import os

from src.task_loaders import *
from data.math.grammar import *

import dreamcoder.domains.list.main as list_legacy

DOMAIN_NAME = "list"
ROOT_DIR = os.getcwd()
DEFAULT_DATA_DIRECTORY = os.path.join(ROOT_DIR, f"dreamcoder/data/")

@TaskLoaderRegistry.register
class ListLoader(TaskDataLoader):
    name = "list"

    def load_tasks(self):

        all_tasks = list_legacy.retrieveJSONTasks(
            filename=os.path.join(DEFAULT_DATA_DIRECTORY, "list_tasks2.json"),
        )[:105]
        
        train_tasks = all_tasks[:80]
        test_tasks = all_tasks[80:]

        return {TRAIN: train_tasks, TEST: test_tasks}
    

@TaskLanguageLoaderRegistry.register
class ListLanguageLoader(TaskDataLoader):
    name = "list"

    def load_task_language(self):
        return ({},{}) # No language for math tasks
    

