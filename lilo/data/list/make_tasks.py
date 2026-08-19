"""
list : make_tasks.py | Author : Sagnik Anupam.

Utility functions for loading tasks for the list domain from DreamCoder.
"""

import os

from src.task_loaders import *

import dreamcoder.domains.list.main as list_legacy

DOMAIN_NAME = "list"
ROOT_DIR = os.getcwd()
DEFAULT_DATA_DIRECTORY = os.path.join(ROOT_DIR, f"dreamcoder/data/")

def check_task_type(request):
    types = str(request).split("->")
    permitted_types = ["int", "list(int)", "bool"]
    for t in types:
        if t.strip() not in permitted_types:
            return False
    return True

@TaskLoaderRegistry.register
class ListLoader(TaskDataLoader):
    name = "list"

    def load_tasks(self):

        all_tasks = list_legacy.retrieveJSONTasks(
            filename=os.path.join(DEFAULT_DATA_DIRECTORY, "list_tasks2.json"),
        )#[:105]
        
        additional_tasks = [x for x in all_tasks if str(x.name) in ["((last (dyn . 0)))", "((len (dyn . 0)))", "((min (dyn . 0)))"]]
        train_tasks = [x for x in all_tasks[:-5] if check_task_type(x.request)] #+ additional_tasks #+ list_legacy.make_list_bootstrap_tasks()
        #test_tasks = all_tasks[20:26]
        test_tasks = [x for x in all_tasks[-5:] if check_task_type(x.request)]
        print(f"Loaded {len(train_tasks)} train tasks and {len(test_tasks)} test tasks for the list domain.")
        return {TRAIN: train_tasks, TEST: test_tasks}
    

@TaskLanguageLoaderRegistry.register
class ListLanguageLoader(TaskDataLoader):
    name = "list"

    def load_task_language(self):
        return ({},{}) # No language for list tasks